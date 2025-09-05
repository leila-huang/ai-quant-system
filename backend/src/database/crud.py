"""
基础CRUD操作

提供通用的数据库增删改查操作，支持复杂查询、分页、排序、事务处理等。
"""

from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from datetime import datetime, date
from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.orm import Session, Query
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger

from backend.src.database.models import Base

# 泛型类型定义
ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")


class CRUDError(Exception):
    """CRUD操作异常"""
    pass


class BaseCRUD(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """基础CRUD操作类"""
    
    def __init__(self, model: Type[ModelType]):
        """
        CRUD对象构造函数
        
        Args:
            model: SQLAlchemy模型类
        """
        self.model = model
    
    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        """根据ID获取单个记录"""
        try:
            return db.query(self.model).filter(self.model.id == id).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.model.__name__} by id {id}: {e}")
            raise CRUDError(f"Failed to get {self.model.__name__}") from e
    
    def get_by_uuid(self, db: Session, uuid: str) -> Optional[ModelType]:
        """根据UUID获取单个记录"""
        try:
            if hasattr(self.model, 'uuid'):
                return db.query(self.model).filter(self.model.uuid == uuid).first()
            else:
                raise AttributeError(f"{self.model.__name__} does not have uuid field")
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.model.__name__} by uuid {uuid}: {e}")
            raise CRUDError(f"Failed to get {self.model.__name__}") from e
    
    def get_multi(
        self, 
        db: Session, 
        *,
        skip: int = 0, 
        limit: int = 100,
        order_by: Optional[str] = None,
        order_desc: bool = False,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        """
        获取多个记录
        
        Args:
            db: 数据库session
            skip: 跳过记录数
            limit: 限制记录数
            order_by: 排序字段
            order_desc: 是否降序
            filters: 过滤条件
        """
        try:
            query = db.query(self.model)
            
            # 应用过滤条件
            if filters:
                query = self._apply_filters(query, filters)
            
            # 应用排序
            if order_by and hasattr(self.model, order_by):
                if order_desc:
                    query = query.order_by(desc(getattr(self.model, order_by)))
                else:
                    query = query.order_by(asc(getattr(self.model, order_by)))
            
            return query.offset(skip).limit(limit).all()
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting multiple {self.model.__name__}: {e}")
            raise CRUDError(f"Failed to get {self.model.__name__} list") from e
    
    def count(self, db: Session, filters: Optional[Dict[str, Any]] = None) -> int:
        """获取记录数量"""
        try:
            query = db.query(func.count(self.model.id))
            
            if filters:
                query = self._apply_filters(query, filters)
            
            return query.scalar()
            
        except SQLAlchemyError as e:
            logger.error(f"Error counting {self.model.__name__}: {e}")
            raise CRUDError(f"Failed to count {self.model.__name__}") from e
    
    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        """创建记录"""
        try:
            if hasattr(obj_in, 'dict'):
                obj_in_data = obj_in.dict()
            elif hasattr(obj_in, '__dict__'):
                obj_in_data = obj_in.__dict__
            else:
                obj_in_data = dict(obj_in)
            
            db_obj = self.model(**obj_in_data)
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            
            logger.info(f"Created {self.model.__name__} with id {db_obj.id}")
            return db_obj
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating {self.model.__name__}: {e}")
            raise CRUDError(f"Failed to create {self.model.__name__}") from e
    
    def create_multi(self, db: Session, *, objs_in: List[CreateSchemaType]) -> List[ModelType]:
        """批量创建记录"""
        try:
            db_objs = []
            for obj_in in objs_in:
                if hasattr(obj_in, 'dict'):
                    obj_in_data = obj_in.dict()
                elif hasattr(obj_in, '__dict__'):
                    obj_in_data = obj_in.__dict__
                else:
                    obj_in_data = dict(obj_in)
                
                db_obj = self.model(**obj_in_data)
                db_objs.append(db_obj)
            
            db.add_all(db_objs)
            db.commit()
            
            for db_obj in db_objs:
                db.refresh(db_obj)
            
            logger.info(f"Created {len(db_objs)} {self.model.__name__} records")
            return db_objs
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error batch creating {self.model.__name__}: {e}")
            raise CRUDError(f"Failed to batch create {self.model.__name__}") from e
    
    def update(
        self, 
        db: Session, 
        *, 
        db_obj: ModelType, 
        obj_in: Union[UpdateSchemaType, Dict[str, Any]]
    ) -> ModelType:
        """更新记录"""
        try:
            if hasattr(obj_in, 'dict'):
                obj_data = obj_in.dict(exclude_unset=True)
            elif isinstance(obj_in, dict):
                obj_data = obj_in
            else:
                obj_data = dict(obj_in)
            
            # 更新字段
            for field, value in obj_data.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)
            
            # 更新updated_at字段
            if hasattr(db_obj, 'updated_at'):
                setattr(db_obj, 'updated_at', datetime.utcnow())
            
            db.commit()
            db.refresh(db_obj)
            
            logger.info(f"Updated {self.model.__name__} with id {db_obj.id}")
            return db_obj
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error updating {self.model.__name__} id {db_obj.id}: {e}")
            raise CRUDError(f"Failed to update {self.model.__name__}") from e
    
    def remove(self, db: Session, *, id: int) -> Optional[ModelType]:
        """删除记录"""
        try:
            obj = db.query(self.model).get(id)
            if obj:
                db.delete(obj)
                db.commit()
                logger.info(f"Removed {self.model.__name__} with id {id}")
            return obj
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error removing {self.model.__name__} id {id}: {e}")
            raise CRUDError(f"Failed to remove {self.model.__name__}") from e
    
    def remove_multi(self, db: Session, *, ids: List[int]) -> int:
        """批量删除记录"""
        try:
            count = db.query(self.model).filter(self.model.id.in_(ids)).count()
            db.query(self.model).filter(self.model.id.in_(ids)).delete(synchronize_session=False)
            db.commit()
            
            logger.info(f"Removed {count} {self.model.__name__} records")
            return count
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error batch removing {self.model.__name__}: {e}")
            raise CRUDError(f"Failed to batch remove {self.model.__name__}") from e
    
    def exists(self, db: Session, id: int) -> bool:
        """检查记录是否存在"""
        try:
            return db.query(self.model.id).filter(self.model.id == id).first() is not None
        except SQLAlchemyError as e:
            logger.error(f"Error checking existence of {self.model.__name__} id {id}: {e}")
            return False
    
    def _apply_filters(self, query: Query, filters: Dict[str, Any]) -> Query:
        """应用过滤条件"""
        for key, value in filters.items():
            if not hasattr(self.model, key):
                continue
                
            field = getattr(self.model, key)
            
            # 处理不同类型的过滤条件
            if isinstance(value, dict):
                # 复杂查询条件
                if 'eq' in value:
                    query = query.filter(field == value['eq'])
                elif 'ne' in value:
                    query = query.filter(field != value['ne'])
                elif 'gt' in value:
                    query = query.filter(field > value['gt'])
                elif 'gte' in value:
                    query = query.filter(field >= value['gte'])
                elif 'lt' in value:
                    query = query.filter(field < value['lt'])
                elif 'lte' in value:
                    query = query.filter(field <= value['lte'])
                elif 'like' in value:
                    query = query.filter(field.like(f"%{value['like']}%"))
                elif 'ilike' in value:
                    query = query.filter(field.ilike(f"%{value['ilike']}%"))
                elif 'in' in value:
                    query = query.filter(field.in_(value['in']))
                elif 'not_in' in value:
                    query = query.filter(~field.in_(value['not_in']))
                elif 'is_null' in value:
                    if value['is_null']:
                        query = query.filter(field.is_(None))
                    else:
                        query = query.filter(field.isnot(None))
                elif 'between' in value:
                    start, end = value['between']
                    query = query.filter(field.between(start, end))
            elif isinstance(value, list):
                # IN查询
                query = query.filter(field.in_(value))
            else:
                # 简单等值查询
                query = query.filter(field == value)
        
        return query


# 分页响应模型
class PaginatedResponse:
    """分页响应"""
    
    def __init__(
        self, 
        items: List[Any], 
        total: int, 
        page: int, 
        page_size: int,
        total_pages: Optional[int] = None
    ):
        self.items = items
        self.total = total
        self.page = page
        self.page_size = page_size
        self.total_pages = total_pages or (total + page_size - 1) // page_size
        self.has_next = page < self.total_pages
        self.has_prev = page > 1


def paginate(
    db: Session,
    query: Query,
    page: int = 1,
    page_size: int = 10,
    max_page_size: int = 100
) -> PaginatedResponse:
    """分页查询通用函数"""
    if page < 1:
        page = 1
    if page_size < 1 or page_size > max_page_size:
        page_size = min(max_page_size, max(1, page_size))
    
    try:
        total = query.count()
        skip = (page - 1) * page_size
        items = query.offset(skip).limit(page_size).all()
        
        return PaginatedResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size
        )
        
    except SQLAlchemyError as e:
        logger.error(f"Error in paginate: {e}")
        raise CRUDError("Failed to paginate query") from e


def execute_raw_query(db: Session, sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """执行原生SQL查询"""
    try:
        if params:
            result = db.execute(text(sql), params)
        else:
            result = db.execute(text(sql))
        
        return result.fetchall()
        
    except SQLAlchemyError as e:
        logger.error(f"Error executing raw query: {e}")
        raise CRUDError("Failed to execute raw query") from e


def bulk_insert_mappings(
    db: Session, 
    model: Type[ModelType], 
    mappings: List[Dict[str, Any]]
) -> int:
    """批量插入数据（高性能）"""
    try:
        db.bulk_insert_mappings(model, mappings)
        db.commit()
        
        count = len(mappings)
        logger.info(f"Bulk inserted {count} {model.__name__} records")
        return count
        
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error bulk inserting {model.__name__}: {e}")
        raise CRUDError(f"Failed to bulk insert {model.__name__}") from e


def bulk_update_mappings(
    db: Session,
    model: Type[ModelType],
    mappings: List[Dict[str, Any]]
) -> int:
    """批量更新数据（高性能）"""
    try:
        db.bulk_update_mappings(model, mappings)
        db.commit()
        
        count = len(mappings)
        logger.info(f"Bulk updated {count} {model.__name__} records")
        return count
        
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Error bulk updating {model.__name__}: {e}")
        raise CRUDError(f"Failed to bulk update {model.__name__}") from e
