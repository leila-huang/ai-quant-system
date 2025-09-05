"""
数据库健康检查和管理工具

提供数据库连接健康检查、性能监控、维护操作等功能。
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from sqlalchemy import text, func
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger

from backend.src.database.connection import get_database_manager
from backend.src.database.models import SystemLog, User, Strategy, Order


class DatabaseHealth:
    """数据库健康检查类"""
    
    def __init__(self):
        self.db_manager = get_database_manager()
    
    def check_connection(self) -> Dict[str, Any]:
        """检查数据库连接状态"""
        start_time = time.time()
        
        try:
            is_connected = self.db_manager.check_connection()
            connection_time = time.time() - start_time
            
            return {
                "status": "healthy" if is_connected else "unhealthy",
                "connected": is_connected,
                "connection_time_ms": round(connection_time * 1000, 2),
                "checked_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "connected": False,
                "error": str(e),
                "connection_time_ms": round((time.time() - start_time) * 1000, 2),
                "checked_at": datetime.utcnow().isoformat()
            }
    
    def check_pool_status(self) -> Dict[str, Any]:
        """检查连接池状态"""
        try:
            pool_info = self.db_manager.get_connection_info()
            
            pool_size = pool_info.get("pool_size", 0)
            checked_out = pool_info.get("checked_out", 0)
            overflow = pool_info.get("overflow", 0)
            
            # 计算连接池使用率
            utilization = (checked_out / max(pool_size, 1)) * 100
            
            status = "healthy"
            if utilization > 90:
                status = "critical"
            elif utilization > 70:
                status = "warning"
            
            return {
                "status": status,
                "pool_size": pool_size,
                "checked_out": checked_out,
                "checked_in": pool_info.get("checked_in", 0),
                "overflow": overflow,
                "utilization_percent": round(utilization, 2),
                "checked_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error checking pool status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
    
    def check_query_performance(self) -> Dict[str, Any]:
        """检查查询性能"""
        performance_tests = []
        
        try:
            with self.db_manager.get_session() as db:
                # 测试简单查询
                start_time = time.time()
                db.execute(text("SELECT 1"))
                simple_query_time = (time.time() - start_time) * 1000
                
                performance_tests.append({
                    "test": "simple_select",
                    "time_ms": round(simple_query_time, 2),
                    "status": "ok" if simple_query_time < 100 else "slow"
                })
                
                # 测试系统表查询
                start_time = time.time()
                db.execute(text("SELECT COUNT(*) FROM information_schema.tables"))
                table_query_time = (time.time() - start_time) * 1000
                
                performance_tests.append({
                    "test": "system_table_query",
                    "time_ms": round(table_query_time, 2),
                    "status": "ok" if table_query_time < 500 else "slow"
                })
                
                # 测试业务表查询（如果表存在）
                try:
                    start_time = time.time()
                    db.query(func.count(User.id)).scalar()
                    user_count_time = (time.time() - start_time) * 1000
                    
                    performance_tests.append({
                        "test": "user_count_query",
                        "time_ms": round(user_count_time, 2),
                        "status": "ok" if user_count_time < 200 else "slow"
                    })
                except SQLAlchemyError:
                    # 表可能不存在，跳过
                    pass
            
            # 计算总体状态
            avg_time = sum(test["time_ms"] for test in performance_tests) / len(performance_tests)
            overall_status = "healthy" if avg_time < 300 else "degraded"
            
            return {
                "status": overall_status,
                "average_query_time_ms": round(avg_time, 2),
                "tests": performance_tests,
                "checked_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking query performance: {e}")
            return {
                "status": "error",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
    
    def check_table_sizes(self) -> Dict[str, Any]:
        """检查表大小和记录数"""
        try:
            table_stats = []
            
            with self.db_manager.get_session() as db:
                # 获取表大小信息（PostgreSQL特有）
                query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        most_common_vals,
                        n_tup_ins,
                        n_tup_upd,
                        n_tup_del,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                    FROM pg_stats 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                    LIMIT 20
                """)
                
                try:
                    result = db.execute(query)
                    for row in result:
                        table_stats.append({
                            "table": row.tablename,
                            "size": row.size,
                            "inserts": row.n_tup_ins,
                            "updates": row.n_tup_upd,
                            "deletes": row.n_tup_del
                        })
                except SQLAlchemyError:
                    # Fallback to basic table counting
                    tables = [User, Strategy, Order, SystemLog]
                    for table in tables:
                        try:
                            count = db.query(func.count(table.id)).scalar()
                            table_stats.append({
                                "table": table.__tablename__,
                                "count": count,
                                "size": "unknown"
                            })
                        except SQLAlchemyError:
                            continue
            
            return {
                "status": "ok",
                "tables": table_stats,
                "checked_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking table sizes: {e}")
            return {
                "status": "error",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
    
    def get_recent_errors(self, hours: int = 24, limit: int = 50) -> Dict[str, Any]:
        """获取最近的错误日志"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            with self.db_manager.get_session() as db:
                errors = db.query(SystemLog).filter(
                    SystemLog.level.in_(["error", "critical"]),
                    SystemLog.created_at >= cutoff_time
                ).order_by(SystemLog.created_at.desc()).limit(limit).all()
                
                error_list = []
                for error in errors:
                    error_list.append({
                        "id": error.id,
                        "level": error.level.value,
                        "module": error.module,
                        "message": error.message,
                        "created_at": error.created_at.isoformat()
                    })
                
                return {
                    "status": "ok",
                    "error_count": len(error_list),
                    "errors": error_list,
                    "time_range_hours": hours,
                    "checked_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting recent errors: {e}")
            return {
                "status": "error",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
    
    def get_full_health_report(self) -> Dict[str, Any]:
        """获取完整的健康检查报告"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy"
        }
        
        # 连接检查
        connection_check = self.check_connection()
        report["connection"] = connection_check
        
        # 连接池检查
        pool_check = self.check_pool_status()
        report["connection_pool"] = pool_check
        
        # 性能检查
        performance_check = self.check_query_performance()
        report["performance"] = performance_check
        
        # 表大小检查
        tables_check = self.check_table_sizes()
        report["tables"] = tables_check
        
        # 错误日志检查
        errors_check = self.get_recent_errors()
        report["recent_errors"] = errors_check
        
        # 确定总体状态
        checks = [connection_check, pool_check, performance_check, tables_check, errors_check]
        statuses = [check.get("status", "unknown") for check in checks]
        
        if "error" in statuses or "critical" in statuses:
            report["overall_status"] = "unhealthy"
        elif "warning" in statuses or "degraded" in statuses:
            report["overall_status"] = "degraded"
        elif "unhealthy" in statuses:
            report["overall_status"] = "unhealthy"
        
        # 添加建议
        report["recommendations"] = self._get_recommendations(report)
        
        return report
    
    def _get_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """基于健康检查结果生成建议"""
        recommendations = []
        
        # 连接相关建议
        if report.get("connection", {}).get("status") != "healthy":
            recommendations.append("检查数据库连接配置和网络连接")
        
        # 连接池相关建议
        pool_status = report.get("connection_pool", {})
        if pool_status.get("utilization_percent", 0) > 80:
            recommendations.append("考虑增加连接池大小或优化长时间运行的查询")
        
        # 性能相关建议
        performance = report.get("performance", {})
        if performance.get("average_query_time_ms", 0) > 500:
            recommendations.append("查询性能较慢，考虑添加索引或优化查询语句")
        
        # 错误相关建议
        error_count = report.get("recent_errors", {}).get("error_count", 0)
        if error_count > 10:
            recommendations.append(f"发现 {error_count} 个最近错误，建议检查应用日志")
        
        if not recommendations:
            recommendations.append("数据库运行状态良好")
        
        return recommendations


def get_health_checker() -> DatabaseHealth:
    """获取数据库健康检查器实例"""
    return DatabaseHealth()
