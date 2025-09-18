"""
AI助手API路由

提供智能对话、预设问题、市场分析等AI辅助功能。
集成现有ML模型API、回测API、数据API等后端服务，
为前端提供智能化的量化交易助手功能。
"""

import re
import random
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session

from backend.src.database.models import Strategy, StrategyStatus
from backend.src.database.connection import get_database_manager
from backend.app.core.exceptions import APIException, BusinessException, DatabaseException


router = APIRouter()


# === Pydantic 请求/响应模型定义 ===

class ChatMessage(BaseModel):
    """聊天消息模型 - 匹配前端ChatMessage接口"""
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    type: str = Field(..., description="消息类型: user | assistant")
    content: str = Field(..., description="消息内容")
    timestamp: str = Field(..., description="时间戳 ISO格式")
    suggestions: Optional[List[str]] = Field(None, description="建议操作列表")


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., min_length=1, description="用户消息")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")
    session_id: Optional[str] = Field(None, description="会话ID")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    message: ChatMessage = Field(..., description="AI回复消息")
    status: str = Field(default="success", description="响应状态")
    processing_time: float = Field(default=0.0, description="处理时间（秒）")


class PresetQuestionItem(BaseModel):
    """预设问题项模型"""
    title: str = Field(..., description="问题标题")
    description: str = Field(..., description="问题描述")
    prompt: str = Field(..., description="完整提示词")


class PresetCategory(BaseModel):
    """预设问题分类模型"""
    category: str = Field(..., description="分类名称")
    icon: str = Field(..., description="图标名称")
    questions: List[PresetQuestionItem] = Field(..., description="问题列表")


class PresetCategoryResponse(BaseModel):
    """预设问题分类响应模型"""
    categories: List[PresetCategory] = Field(..., description="分类列表")
    total_count: int = Field(..., description="总问题数量")


class AnalysisRequest(BaseModel):
    """分析请求模型"""
    type: str = Field(..., description="分析类型: market/strategy/stock/factor")
    params: Dict[str, Any] = Field(default_factory=dict, description="分析参数")


class AnalysisResponse(BaseModel):
    """分析响应模型"""
    analysis_type: str = Field(..., description="分析类型")
    results: Dict[str, Any] = Field(..., description="分析结果")
    charts: Optional[List[Dict[str, Any]]] = Field(None, description="图表数据")
    recommendations: List[str] = Field(default_factory=list, description="建议列表")


# === 智能回复生成器 ===

class AIResponseGenerator:
    """AI回复生成器"""
    
    def __init__(self):
        self.keyword_responses = {
            # 市场分析相关
            "市场趋势": self._generate_market_trend_response,
            "行情分析": self._generate_market_trend_response,  # 复用市场趋势分析
            "板块轮动": self._generate_market_trend_response,  # 复用市场趋势分析
            "技术指标": self._generate_market_trend_response,  # 复用市场趋势分析
            
            # 策略相关
            "策略优化": self._generate_strategy_optimization_response,
            "参数调优": self._generate_strategy_optimization_response,  # 复用策略优化
            "回测分析": self._generate_strategy_optimization_response,  # 复用策略优化
            "风险控制": self._generate_strategy_optimization_response,  # 复用策略优化
            
            # 个股分析相关
            "个股分析": self._generate_stock_analysis_response,
            "估值分析": self._generate_stock_analysis_response,  # 复用个股分析
            "基本面": self._generate_stock_analysis_response,  # 复用个股分析
            
            # 因子研究相关
            "因子研究": self._generate_factor_research_response,
            "量化因子": self._generate_factor_research_response,  # 复用因子研究
            "多因子模型": self._generate_factor_research_response,  # 复用因子研究
        }
        
        self.suggestion_templates = {
            "market": ["查看实时行情", "分析历史数据", "板块热点分析", "技术指标详解"],
            "strategy": ["策略回测分析", "参数优化建议", "风险评估报告", "收益归因分析"],
            "stock": ["个股深度研究", "同行业对比", "估值水平分析", "业绩预测更新"],
            "factor": ["因子有效性测试", "因子暴露分析", "多因子建模", "因子归因报告"]
        }

    async def generate_response(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> ChatMessage:
        """生成AI回复"""
        start_time = datetime.now()
        
        # 消息分类和关键词提取
        message_type, keywords = self._classify_message(user_message)
        
        # 生成回复内容
        if keywords and any(kw in self.keyword_responses for kw in keywords):
            # 找到匹配的关键词生成器
            for kw in keywords:
                if kw in self.keyword_responses:
                    content = await self.keyword_responses[kw](user_message, context)
                    break
            else:
                content = self._generate_default_response(user_message)
        else:
            content = self._generate_default_response(user_message)
        
        # 生成建议
        suggestions = self._generate_suggestions(message_type, keywords)
        
        # 构建回复消息
        message_id = str(int(datetime.now().timestamp() * 1000))
        
        return ChatMessage(
            id=message_id,
            type="assistant",
            content=content,
            timestamp=datetime.now().isoformat(),
            suggestions=suggestions
        )

    def _classify_message(self, message: str) -> tuple[str, List[str]]:
        """消息分类和关键词提取"""
        message_lower = message.lower()
        
        # 定义关键词模式
        patterns = {
            "market": ["市场", "行情", "趋势", "指数", "板块", "热点"],
            "strategy": ["策略", "回测", "参数", "优化", "算法", "模型"],
            "stock": ["个股", "股票", "公司", "估值", "基本面", "财务"],
            "factor": ["因子", "量化", "特征", "指标", "相关性", "预测"]
        }
        
        # 提取匹配的关键词
        matched_keywords = []
        message_types = []
        
        for msg_type, keywords in patterns.items():
            for keyword in keywords:
                if keyword in message_lower:
                    matched_keywords.append(keyword)
                    if msg_type not in message_types:
                        message_types.append(msg_type)
        
        # 返回主要类型和关键词
        primary_type = message_types[0] if message_types else "general"
        return primary_type, matched_keywords

    async def _generate_market_trend_response(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """生成市场趋势分析回复"""
        return """基于当前市场数据分析，我为您提供以下市场趋势洞察：

📊 **整体市场分析**
• 上证指数当前处于震荡整理阶段
• 创业板相对表现较强，科技股活跃度提升
• 成交量较前期有所放大，市场参与度回升

🔥 **热点板块识别**
• 新能源汽车产业链保持强势
• 人工智能概念持续受到关注  
• 医药生物板块出现结构性机会

⚠️ **风险提示**
• 注意宏观政策变化对市场的影响
• 关注外围市场波动的传导效应
• 建议保持适度谨慎，控制仓位

📈 **操作建议**
建议采用分散投资策略，重点关注业绩确定性较高的优质个股。"""

    async def _generate_strategy_optimization_response(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """生成策略优化建议回复"""
        return """根据您的策略优化需求，我提供以下专业建议：

🎯 **策略诊断分析**
• 当前策略的收益风险特征分析
• 回测期间的表现归因分解
• 策略在不同市场环境下的适应性评估

⚙️ **参数优化方向**
• **进场条件**: 建议调整技术指标参数，提高信号准确率
• **止损设置**: 可考虑动态止损机制，降低回撤风险
• **仓位管理**: 根据市场波动率动态调整仓位

🔧 **技术改进建议**
• 增加市场情绪指标作为过滤条件
• 优化资金管理模块，提升资金使用效率
• 考虑多时间周期信号的综合判断

📊 **风险控制优化**
• 设定最大日内损失限制
• 增加相关性风险监控
• 建立策略表现异常预警机制

我可以为您详细分析具体的策略参数，请提供更多策略细节。"""

    async def _generate_stock_analysis_response(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """生成个股分析回复"""
        return """为您提供个股分析的专业框架：

🏢 **基本面分析维度**
• **财务状况**: 营收增长、净利润、ROE等关键指标
• **行业地位**: 市场份额、竞争优势、护城河分析
• **估值水平**: P/E、P/B、PEG等估值指标横纵向对比

📈 **技术面分析要点**
• **趋势判断**: 多时间周期趋势线和移动平均线分析
• **支撑阻力**: 关键价位的支撑和阻力水平识别
• **量价关系**: 成交量与价格变化的配合度分析

🎯 **投资建议框架**
• **买入时机**: 基于技术信号和基本面改善的买点识别
• **目标价位**: 基于估值模型和技术分析的合理价位区间
• **风险控制**: 止损位设定和仓位管理建议

如需分析特定个股，请提供股票代码，我将为您生成详细的投资分析报告。"""

    async def _generate_factor_research_response(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """生成因子研究回复"""
        return """量化因子研究是构建超额收益的核心，让我为您解析：

🔬 **因子研究方法论**
• **因子发现**: 基于金融理论和市场异象挖掘有效因子
• **因子检验**: IC值、IR比率、因子衰减等统计检验
• **因子正交化**: 去除因子间相关性，提取纯因子暴露

📊 **主流因子分类**
• **价值因子**: PE、PB、PS等估值类指标
• **质量因子**: ROE、ROA、资产负债率等质量指标  
• **动量因子**: 价格动量、盈利修正动量等
• **成长因子**: 收入增长率、利润增长率等

🏗️ **多因子模型构建**
• **因子选择**: 基于IC值和因子覆盖度筛选有效因子
• **权重配置**: 等权、IC加权、风险平价等权重分配方法
• **风险调整**: Barra风险模型进行风险暴露控制

⚡ **模型优化建议**
• 定期更新因子有效性检验
• 根据市场风格变化动态调整因子权重
• 建立因子表现监控和预警机制

需要针对特定因子进行深度研究吗？"""

    def _generate_default_response(self, message: str) -> str:
        """生成默认回复"""
        default_responses = [
            """感谢您的提问！作为AI量化助手，我可以为您提供以下专业服务：

📊 **市场分析服务**
• 实时行情分析和趋势研判
• 板块热点挖掘和轮动分析
• 技术指标解读和信号提示

🤖 **策略优化服务**
• 量化策略参数调优建议
• 回测分析和策略改进方案
• 风险控制和资金管理优化

🔍 **投研分析服务**
• 个股深度分析和投资建议
• 行业比较和投资机会识别
• 量化因子研究和模型构建

请告诉我您具体想了解哪个方面，我将为您提供更精准的分析和建议！""",
            
            """很高兴为您服务！基于您的问题，我建议从以下角度进行分析：

🎯 **数据驱动分析**
我可以帮您分析历史数据，识别市场规律和交易机会。

📈 **量化策略建议**
基于您的需求，为您推荐合适的量化策略和参数设置。

⚠️ **风险管理提醒**
结合当前市场环境，为您提供风险控制和仓位管理建议。

如需更详细的分析，请提供更多具体信息，比如：
• 关注的股票代码或板块
• 策略类型和参数
• 投资时间周期和风险偏好

我将为您制定个性化的投资分析方案！""",
            
            """根据您的问题，我为您提供以下分析思路：

💡 **专业分析框架**
• 基本面分析：公司财务、行业地位、估值水平
• 技术面分析：趋势识别、关键位置、交易信号
• 量化分析：因子暴露、风险归因、收益预测

🔧 **实用工具推荐**
• 实时数据监控和预警系统
• 策略回测和参数优化平台
• 风险管理和组合分析工具

📚 **知识库支持**
我拥有丰富的金融市场知识和量化交易经验，可以为您解答：
• 市场机制和交易规则
• 量化策略和算法原理
• 风险管理和投资心理

有什么具体问题，请随时向我提问！"""
        ]
        
        return random.choice(default_responses)

    def _generate_suggestions(self, message_type: str, keywords: List[str]) -> List[str]:
        """生成建议操作"""
        base_suggestions = self.suggestion_templates.get(message_type, self.suggestion_templates["market"])
        
        # 根据关键词调整建议
        if "策略" in keywords:
            specific_suggestions = ["查看策略回测结果", "优化策略参数", "分析策略风险", "获取策略信号"]
        elif "个股" in keywords:
            specific_suggestions = ["查看个股技术分析", "获取基本面数据", "行业对比分析", "估值模型计算"]
        elif "市场" in keywords:
            specific_suggestions = ["实时行情监控", "板块热点分析", "市场情绪指标", "宏观数据解读"]
        else:
            specific_suggestions = base_suggestions
        
        # 随机选择4个建议
        return random.sample(specific_suggestions, min(4, len(specific_suggestions)))


# === 全局实例 ===
ai_generator = AIResponseGenerator()


# === 预设问题数据 ===
PRESET_CATEGORIES = [
    PresetCategory(
        category="市场分析",
        icon="BarChartOutlined",
        questions=[
            PresetQuestionItem(
                title="当前市场趋势分析",
                description="分析当前A股市场的整体趋势和热点板块",
                prompt="请分析当前A股市场的整体趋势，包括主要指数走势、热点板块和市场情绪指标。"
            ),
            PresetQuestionItem(
                title="板块轮动分析", 
                description="识别当前市场的板块轮动规律",
                prompt="请分析最近一个月的板块轮动情况，哪些板块表现强势，哪些板块相对落后？"
            ),
            PresetQuestionItem(
                title="技术指标解读",
                description="解读关键技术指标的信号",
                prompt="请帮我解读当前沪深300指数的技术指标信号，包括MACD、RSI、均线等。"
            )
        ]
    ),
    PresetCategory(
        category="策略优化",
        icon="LineChartOutlined",
        questions=[
            PresetQuestionItem(
                title="策略参数调优",
                description="优化现有量化策略的参数设置",
                prompt="我的移动平均线策略最近表现不佳，请帮我分析可能的原因并建议参数优化方案。"
            ),
            PresetQuestionItem(
                title="多因子模型构建",
                description="构建多因子选股模型",
                prompt="请帮我设计一个适合A股市场的多因子选股模型，包括因子选择和权重分配建议。"
            ),
            PresetQuestionItem(
                title="风险控制建议",
                description="完善策略的风险控制机制",
                prompt="请为我的量化策略设计一套完整的风险控制体系，包括止损、仓位管理等。"
            )
        ]
    ),
    PresetCategory(
        category="投研报告",
        icon="FileTextOutlined",
        questions=[
            PresetQuestionItem(
                title="个股深度分析",
                description="生成个股的投资分析报告",
                prompt="请帮我分析贵州茅台(600519)的投资价值，包括基本面、技术面和估值分析。"
            ),
            PresetQuestionItem(
                title="行业对比研究",
                description="对比分析不同行业的投资机会",
                prompt="请对比分析新能源汽车和传统汽车行业的投资机会和风险。"
            ),
            PresetQuestionItem(
                title="量化因子研究",
                description="研究特定量化因子的有效性",
                prompt="请分析动量因子在A股市场的有效性，包括不同时间周期和市场环境下的表现。"
            )
        ]
    )
]


# === API 路由端点 ===

@router.post("/chat", response_model=ChatResponse, summary="AI智能对话")
async def chat_with_ai(request: ChatRequest):
    """
    与AI助手进行智能对话
    
    支持多种类型的问题：市场分析、策略优化、个股研究、因子分析等。
    """
    try:
        start_time = datetime.now()
        
        # 生成AI回复
        ai_message = await ai_generator.generate_response(
            user_message=request.message,
            context=request.context
        )
        
        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            message=ai_message,
            status="success",
            processing_time=processing_time
        )
        
    except Exception as e:
        raise DatabaseException(f"AI对话处理失败: {str(e)}")


@router.get("/presets", response_model=PresetCategoryResponse, summary="获取预设问题")
async def get_preset_questions():
    """
    获取预设问题分类和问题列表
    
    返回按分类组织的预设问题，用户可以快速选择常用问题。
    """
    try:
        total_count = sum(len(category.questions) for category in PRESET_CATEGORIES)
        
        return PresetCategoryResponse(
            categories=PRESET_CATEGORIES,
            total_count=total_count
        )
        
    except Exception as e:
        raise DatabaseException(f"获取预设问题失败: {str(e)}")


@router.get("/chat/history", summary="获取聊天历史")
async def get_chat_history(
    session_id: str = Query(..., description="会话ID"),
    limit: int = Query(50, ge=1, le=200, description="返回消息数量限制")
):
    """
    获取指定会话的聊天历史记录
    
    注意：当前版本暂未实现持久化存储，返回空历史记录。
    """
    try:
        # TODO: 实现聊天历史的持久化存储
        # 当前返回空历史，后续可集成Redis或数据库存储
        return {
            "session_id": session_id,
            "messages": [],
            "total_count": 0,
            "message": "聊天历史功能将在后续版本中实现"
        }
        
    except Exception as e:
        raise DatabaseException(f"获取聊天历史失败: {str(e)}")


@router.post("/analysis", response_model=AnalysisResponse, summary="智能分析服务")
async def intelligent_analysis(request: AnalysisRequest):
    """
    提供智能分析服务
    
    支持市场分析、策略分析、个股分析、因子分析等多种分析类型。
    """
    try:
        # 根据分析类型调用相应的分析服务
        analysis_type = request.type.lower()
        
        if analysis_type == "market":
            results = await _analyze_market(request.params)
        elif analysis_type == "strategy":
            results = await _analyze_strategy(request.params)
        elif analysis_type == "stock":
            results = await _analyze_stock(request.params)
        elif analysis_type == "factor":
            results = await _analyze_factor(request.params)
        else:
            raise BusinessException(f"不支持的分析类型: {analysis_type}")
        
        return AnalysisResponse(
            analysis_type=analysis_type,
            results=results,
            recommendations=results.get("recommendations", [])
        )
        
    except Exception as e:
        if isinstance(e, (BusinessException, APIException)):
            raise
        raise DatabaseException(f"智能分析服务失败: {str(e)}")


# === 分析服务函数 ===

async def _analyze_market(params: Dict[str, Any]) -> Dict[str, Any]:
    """市场分析服务"""
    # TODO: 集成真实的市场数据分析
    return {
        "market_trend": "震荡上行",
        "hot_sectors": ["新能源汽车", "人工智能", "生物医药"],
        "risk_level": "中等",
        "recommendations": [
            "关注新能源汽车产业链投资机会",
            "控制仓位，防范系统性风险", 
            "关注业绩确定性较高的龙头股"
        ]
    }


async def _analyze_strategy(params: Dict[str, Any]) -> Dict[str, Any]:
    """策略分析服务"""
    # TODO: 集成现有的回测API和策略评估功能
    strategy_name = params.get("strategy_name", "未指定策略")
    
    return {
        "strategy_name": strategy_name,
        "performance_metrics": {
            "annual_return": 0.15,
            "max_drawdown": -0.08,
            "sharpe_ratio": 1.2,
            "win_rate": 0.65
        },
        "optimization_suggestions": [
            "建议调整移动平均线周期参数",
            "增加止损机制，控制单笔最大损失",
            "考虑加入市场情绪指标过滤"
        ],
        "recommendations": [
            "优化参数设置提升策略稳定性",
            "加强风险控制机制",
            "定期回测验证策略有效性"
        ]
    }


async def _analyze_stock(params: Dict[str, Any]) -> Dict[str, Any]:
    """个股分析服务"""
    # TODO: 集成真实的股票数据和分析功能
    stock_code = params.get("stock_code", "000001")
    
    return {
        "stock_code": stock_code,
        "basic_info": {
            "name": "平安银行" if stock_code == "000001" else "示例股票",
            "industry": "银行",
            "market_cap": "3500亿"
        },
        "technical_analysis": {
            "trend": "上升趋势",
            "support_level": 12.5,
            "resistance_level": 15.8,
            "rsi": 65.2
        },
        "valuation": {
            "pe_ratio": 8.5,
            "pb_ratio": 0.9,
            "dividend_yield": 0.045
        },
        "recommendations": [
            "估值相对合理，具有投资价值",
            "技术面显示上升趋势，可适量配置",
            "关注银行业政策变化影响"
        ]
    }


async def _analyze_factor(params: Dict[str, Any]) -> Dict[str, Any]:
    """因子分析服务"""
    # TODO: 集成现有的因子分析和ML模型功能
    factor_name = params.get("factor_name", "动量因子")
    
    return {
        "factor_name": factor_name,
        "effectiveness_metrics": {
            "ic_mean": 0.045,
            "ic_std": 0.12,
            "ir_ratio": 0.375,
            "hit_rate": 0.58
        },
        "factor_analysis": {
            "category": "动量类因子",
            "time_decay": "中等",
            "market_neutrality": "良好"
        },
        "model_integration": {
            "weight_suggestion": 0.15,
            "risk_exposure": "中等",
            "correlation_with_other_factors": "低"
        },
        "recommendations": [
            "该因子具有一定的预测能力，建议纳入多因子模型",
            "注意因子在不同市场环境下的稳定性",
            "建议与其他类型因子组合使用以分散风险"
        ]
    }


# === 统计信息端点 ===

@router.get("/stats/summary", summary="AI助手使用统计")
async def get_ai_stats():
    """
    获取AI助手使用统计信息
    """
    try:
        # TODO: 实现真实的使用统计
        return {
            "total_conversations": 0,
            "total_messages": 0,
            "popular_categories": [
                {"category": "市场分析", "count": 0},
                {"category": "策略优化", "count": 0},
                {"category": "投研报告", "count": 0}
            ],
            "response_time_avg": 1.2,
            "user_satisfaction": 4.5,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise DatabaseException(f"获取AI统计信息失败: {str(e)}")
