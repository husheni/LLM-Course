#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于阿里云通义千问的RAG系统实现
支持多层级文本切片、语义检索和相似度计算
"""

import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging

# 尝试导入阿里云相关组件
try:
    import dashscope
    # 直接使用 dashscope 而不是 langchain 包装器
    ALIBABA_CLOUD_AVAILABLE = True
except ImportError:
    ALIBABA_CLOUD_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashScopeEmbeddingWrapper:
    """DashScope Embedding 封装类"""
    
    def __init__(self, model: str = "text-embedding-v1"):
        self.model = model
        self.dashscope = dashscope
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量获取文档embedding"""
        try:
            response = self.dashscope.TextEmbedding.call(
                model=self.model,
                input=texts,
                text_type="document"
            )
            if response and hasattr(response, 'output') and hasattr(response.output, 'embeddings'):
                embeddings = []
                for item in response.output.embeddings:
                    if hasattr(item, 'embedding'):
                        embeddings.append(item.embedding)
                return embeddings
            else:
                logger.error("获取embedding失败")
                return []
        except Exception as e:
            logger.error(f"DashScope embedding调用失败: {e}")
            return []
    
    def embed_query(self, text: str) -> List[float]:
        """获取查询embedding"""
        try:
            response = self.dashscope.TextEmbedding.call(
                model=self.model,
                input=[text],
                text_type="query"
            )
            if response and hasattr(response, 'output') and hasattr(response.output, 'embeddings'):
                if response.output.embeddings and len(response.output.embeddings) > 0:
                    return response.output.embeddings[0].embedding
            logger.error("获取query embedding失败")
            return []
        except Exception as e:
            logger.error(f"DashScope query embedding调用失败: {e}")
            return []

@dataclass
class RetrievalConfig:
    """检索配置类"""
    title_weight: float = 0.6
    question_weight: float = 0.4
    answer_weight: float = 0.0
    
    def normalize_weights(self):
        """标准化权重"""
        total = self.title_weight + self.question_weight + self.answer_weight
        if total > 0:
            self.title_weight /= total
            self.question_weight /= total
            self.answer_weight /= total

class TextChunk:
    """文本块类"""
    def __init__(self, content: str, chunk_type: str, original_index: int):
        self.content = content
        self.chunk_type = chunk_type  # 'title', 'question', 'answer'
        self.original_index = original_index

class DataRecord:
    """数据记录类"""
    def __init__(self, index: int, questionTitle: str, questionText: str, answerText: str):
        self.index = index
        self.questionTitle = questionTitle
        self.questionText = questionText
        self.answerText = answerText
        self.questionTitle_chunks = []
        self.questionText_chunks = []
        self.answerText_chunks = []

class QwenRAGSystem:
    """基于通义千问的RAG系统"""
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-v1",
        llm_model: str = "qwen-turbo",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        retrieval_config: Optional[RetrievalConfig] = None
    ):
        """
        初始化RAG系统
        
        Args:
            embedding_model: embedding模型名称
            llm_model: LLM模型名称
            chunk_size: 文本切片大小
            chunk_overlap: 文本重叠大小
            retrieval_config: 检索配置
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化配置
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.retrieval_config.normalize_weights()
        
        # 初始化组件
        self._init_components()
        
        # 数据存储
        self.data_records: List[DataRecord] = []
        self.vectorstore = None
        
    def _init_components(self):
        """初始化组件"""
        try:
            # 检查阿里云配置
            self._check_alibaba_cloud_config()
            
            # 初始化embedding和llm - 专门为阿里云通义千问设计
            if not ALIBABA_CLOUD_AVAILABLE:
                raise RuntimeError("阿里云SDK不可用，无法初始化通义千问RAG系统")
            
            # 使用 DashScope embedding wrapper
            self.embeddings = DashScopeEmbeddingWrapper(model=self.embedding_model)
            self.dashscope = dashscope
            logger.info(f"使用阿里云通义千问模型: {self.llm_model}")
            
            # 初始化文本分割器
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            logger.info("组件初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    
    def _check_alibaba_cloud_config(self):
        """检查阿里云配置"""
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            logger.warning("未设置DASHSCOPE_API_KEY环境变量")
            if not ALIBABA_CLOUD_AVAILABLE:
                raise ValueError("需要设置阿里云API密钥或安装相关SDK")
        
        # 检查region
        region = os.getenv('ALIBABA_CLOUD_REGION', 'cn-beijing')
        logger.info(f"使用阿里云区域: {region}")
    
    def _semantic_chunk_text(self, text: str, chunk_type: str, record_index: int) -> List[TextChunk]:
        """语义切片文本"""
        if not text or text.strip() == "":
            return [TextChunk("", chunk_type, record_index)]
        
        try:
            # 使用langchain的文本分割器
            splits = self.text_splitter.split_text(text)
            return [TextChunk(split, chunk_type, record_index) for split in splits]
        except Exception as e:
            logger.warning(f"文本切片失败: {e}, 使用原始文本")
            return [TextChunk(text, chunk_type, record_index)]
    
    def _create_embeddings_for_chunks(self, chunks: List[TextChunk]) -> List[np.ndarray]:
        """为文本块创建embedding"""
        try:
            texts = [chunk.content for chunk in chunks]
            embeddings = self.embeddings.embed_documents(texts)
            return [np.array(emb) for emb in embeddings]
        except Exception as e:
            logger.error(f"创建embedding失败: {e}")
            return []
    
    def load_and_process_data(self, data_path: str) -> bool:
        """
        加载和处理数据
        
        Args:
            data_path: JSON数据文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 加载JSON数据
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            logger.info(f"加载了 {len(raw_data)} 条原始数据")
            
            # 处理每条记录
            for i, item in enumerate(raw_data):
                try:
                    # 创建数据记录
                    record = DataRecord(
                        index=i,
                        questionTitle=item.get('questionTitle', ''),
                        questionText=item.get('questionText', ''),
                        answerText=item.get('answerText', '')
                    )
                    
                    # 语义切片
                    record.questionTitle_chunks = self._semantic_chunk_text(
                        record.questionTitle, 'title', i
                    )
                    record.questionText_chunks = self._semantic_chunk_text(
                        record.questionText, 'question', i
                    )
                    record.answerText_chunks = self._semantic_chunk_text(
                        record.answerText, 'answer', i
                    )
                    
                    self.data_records.append(record)
                    
                except Exception as e:
                    logger.warning(f"处理第 {i} 条记录失败: {e}")
                    continue
            
            logger.info(f"成功处理了 {len(self.data_records)} 条记录")
            return True
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return False
    
    def build_vectorstore(self) -> bool:
        """
        构建向量数据库
        
        Returns:
            bool: 是否成功
        """
        try:
            all_chunks = []
            chunk_to_record_map = {}
            
            # 收集所有文本块
            for record in self.data_records:
                for chunk in record.questionTitle_chunks + record.questionText_chunks + record.answerText_chunks:
                    if chunk.content.strip():  # 只处理非空内容
                        all_chunks.append(chunk.content)
                        chunk_to_record_map[chunk.content] = {
                            'record': record,
                            'chunk_type': chunk.chunk_type,
                            'chunk_index': len(all_chunks) - 1
                        }
            
            if not all_chunks:
                logger.error("没有有效的文本块用于构建向量数据库")
                return False
            
            # 创建embedding并构建向量数据库
            logger.info(f"为 {len(all_chunks)} 个文本块创建embedding...")
            embeddings = self.embeddings.embed_documents(all_chunks)
            
            # 创建FAISS向量数据库
            # 1. 先构造 text_embeddings（用你已有的 all_chunks 和 embeddings）
            embedding_model = DashScopeEmbeddings(model="text-embedding-v1")
            zero_vector = [0.0] * 768  # 通义千问嵌入向量维度为768，按需调整

            # 第二步：替换空文本对应的向量
            processed_embeddings = []
            for text, emb in zip(all_chunks, embeddings):
                if not text.strip():
                    processed_embeddings.append(zero_vector)
                else:
                    processed_embeddings.append(emb)

            # 第三步：后续逻辑不变（构造 text_embeddings 并调用）
            text_embeddings = list(zip(all_chunks, processed_embeddings))
            if not text_embeddings:
                text_embeddings=zero_vector

            self.vectorstore = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                embedding=embedding_model,
                metadatas=[{
                    'record_index': chunk_to_record_map[text]['record'].index,
                    'chunk_type': chunk_to_record_map[text]['chunk_type'],
                    'original_text': text[:100] + '...' if len(text) > 100 else text
                } for text in all_chunks]
            )

            
            logger.info("向量数据库构建完成")
            return True
            
        except Exception as e:
            logger.error(f"向量数据库构建失败: {e}")
            return False
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot_product / (norm1 * norm2))
        except:
            return 0.0
    
    def _query_rewrite(self, query: str) -> str:
        """
        查询重写接口（可扩展）
        
        Args:
            query: 原始查询
            
        Returns:
            str: 重写后的查询
        """
        # 这里可以实现查询扩展、同义词替换等逻辑
        # 当前直接返回原始查询
        return query.strip()
    
    def _adaptive_similarity_weights(self, query: str, record: DataRecord) -> Tuple[float, float, float]:
        """
        自适应相似度权重接口（可扩展）
        
        Args:
            query: 查询文本
            record: 数据记录
            
        Returns:
            Tuple[float, float, float]: 自适应的权重
        """
        # 这里可以根据查询特征动态调整权重
        # 当前返回默认权重
        return self.retrieval_config.title_weight, \
               self.retrieval_config.question_weight, \
               self.retrieval_config.answer_weight
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        if not self.vectorstore:
            logger.error("向量数据库未构建，请先调用build_vectorstore()")
            return []
        
        try:
            # 查询重写
            rewritten_query = self._query_rewrite(query)
            logger.info(f"原始查询: {query}")
            logger.info(f"重写查询: {rewritten_query}")
            
            # 为查询创建embedding
            query_embedding = np.array(self.embeddings.embed_query(rewritten_query))
            
            # 获取所有向量和元数据
            all_embeddings = self.vectorstore.index.reconstruct_n(0, self.vectorstore.index.ntotal)
            metadatas = self.vectorstore.metadatas
            
            # 计算相似度
            record_scores = {}
            
            for i, metadata in enumerate(metadatas):
                record_index = metadata['record_index']
                chunk_type = metadata['chunk_type']
                
                if record_index not in record_scores:
                    record_scores[record_index] = {
                        'record': None,
                        'title_similarities': [],
                        'question_similarities': [],
                        'answer_similarities': []
                    }
                
                # 计算相似度
                doc_embedding = np.array(all_embeddings[i])
                similarity = self._calculate_cosine_similarity(query_embedding, doc_embedding)
                
                # 分类存储相似度
                if chunk_type == 'title':
                    record_scores[record_index]['title_similarities'].append(similarity)
                elif chunk_type == 'question':
                    record_scores[record_index]['question_similarities'].append(similarity)
                elif chunk_type == 'answer':
                    record_scores[record_index]['answer_similarities'].append(similarity)
            
            # 计算每条记录的最终相似度
            final_scores = []
            
            for record_index, scores_data in record_scores.items():
                record = self.data_records[record_index]
                
                # 计算各部分的平均相似度
                title_avg = np.mean(scores_data['title_similarities']) if scores_data['title_similarities'] else 0.0
                question_avg = np.mean(scores_data['question_similarities']) if scores_data['question_similarities'] else 0.0
                answer_avg = np.mean(scores_data['answer_similarities']) if scores_data['answer_similarities'] else 0.0
                
                # 获取自适应权重
                title_weight, question_weight, answer_weight = self._adaptive_similarity_weights(query, record)
                
                # 计算最终相似度
                final_similarity = (
                    title_weight * title_avg +
                    question_weight * question_avg +
                    answer_weight * answer_avg
                )
                
                final_scores.append({
                    'record_index': record_index,
                    'record': record,
                    'final_similarity': final_similarity,
                    'title_similarity': title_avg,
                    'question_similarity': question_avg,
                    'answer_similarity': answer_avg,
                    'title_weight': title_weight,
                    'question_weight': question_weight,
                    'answer_weight': answer_weight
                })
            
            # 排序并返回前k个结果
            final_scores.sort(key=lambda x: x['final_similarity'], reverse=True)
            return final_scores[:top_k]
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def generate_response(self, query: str, top_k: int = 5) -> str:
        """
        生成回答
        
        Args:
            query: 查询文本
            top_k: 使用的候选文档数量
            
        Returns:
            str: 生成的回答
        """
        try:
            # 检索相关文档
            search_results = self.search(query, top_k)
            
            if not search_results:
                return "抱歉，没有找到相关的回答。"
            
            # 构建上下文
            context_parts = []
            for i, result in enumerate(search_results, 1):
                record = result['record']
                context_parts.append(f"候选 {i} (相似度: {result['final_similarity']:.3f}):")
                context_parts.append(f"问题: {record.questionTitle}")
                if record.questionText:
                    context_parts.append(f"问题详情: {record.questionText}")
                context_parts.append(f"回答: {record.answerText}")
                context_parts.append("-" * 50)
            
            context = "\n".join(context_parts)
            
            # 构建提示词
            prompt = f"""基于以下上下文信息，回答用户的问题。请根据上下文内容提供准确、有用的回答。

上下文信息:
{context}

用户问题: {query}

请基于上述上下文信息回答用户的问题。如果上下文信息不足，请说明并提供一般性的建议。"""
            
            # 生成回答
            if ALIBABA_CLOUD_AVAILABLE and hasattr(self, 'dashscope'):
                try:
                    # 使用DashScope原生API生成回答
                    response = self.dashscope.Generation.call(
                        model=self.llm_model,
                        input={'messages': [{'role': 'user', 'content': prompt}]},
                        temperature=0.1,
                        max_tokens=2000
                    )
                    if response and hasattr(response, 'output') and hasattr(response.output, 'text'):
                        return response.output.text.strip()
                    else:
                        return "抱歉，模型生成失败"
                except Exception as e:
                    logger.error(f"DashScope调用失败: {e}")
                    return f"抱歉，模型调用失败: {e}"
            else:
                # 如果没有LLM，返回基于检索结果的摘要
                best_result = search_results[0]
                return f"基于相似度最高的回答：\n{best_result['record'].answerText}"
            
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return f"抱歉，回答生成失败: {e}"
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        if not self.data_records:
            return {}
        
        total_records = len(self.data_records)
        total_chunks = sum(
            len(record.questionTitle_chunks) + 
            len(record.questionText_chunks) + 
            len(record.answerText_chunks)
            for record in self.data_records
        )
        
        return {
            'total_records': total_records,
            'total_chunks': total_chunks,
            'avg_chunks_per_record': total_chunks / total_records if total_records > 0 else 0,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'retrieval_config': {
                'title_weight': self.retrieval_config.title_weight,
                'question_weight': self.retrieval_config.question_weight,
                'answer_weight': self.retrieval_config.answer_weight
            },
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'using_alibaba_cloud': ALIBABA_CLOUD_AVAILABLE
        }

def main():
    """主函数示例"""
    # 创建RAG系统
    rag = QwenRAGSystem(
        embedding_model="text-embedding-v1",
        llm_model="qwen-turbo",
        chunk_size=500,
        chunk_overlap=100,
        retrieval_config=RetrievalConfig(title_weight=0.6, question_weight=0.4, answer_weight=0.0)
    )
    
    # 数据路径
    data_path = r"E:\1__xubin_hu\Program and setting\datasets\Mental_Health_conv\cl_output_file.json"
    
    # 加载和处理数据
    print("正在加载和处理数据...")
    if not rag.load_and_process_data(data_path):
        print("数据加载失败")
        return
    
    # 构建向量数据库
    print("正在构建向量数据库...")
    if not rag.build_vectorstore():
        print("向量数据库构建失败")
        return
    
    # 显示统计信息
    stats = rag.get_retrieval_stats()
    print("\n系统统计信息:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 搜索测试
    query = "什么是恐慌发作？"
    print(f"\n测试查询: {query}")
    
    # 检索相关文档
    search_results = rag.search(query, top_k=3)
    print(f"\n找到 {len(search_results)} 个相关结果:")
    for i, result in enumerate(search_results, 1):
        print(f"\n结果 {i}:")
        print(f"相似度: {result['final_similarity']:.3f}")
        print(f"问题: {result['record'].questionTitle}")
        print(f"回答预览: {result['record'].answerText[:100]}...")
    
    # 生成完整回答
    print("\n" + "="*80)
    print("生成的回答:")
    print("="*80)
    response = rag.generate_response(query, top_k=3)
    print(response)

if __name__ == "__main__":
    main()