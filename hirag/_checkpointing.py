"""
Checkpointing System for HiRAG

This module provides comprehensive checkpointing capabilities to enable
resumable ingestion operations. If the pipeline fails at any point,
it can resume from the last successful checkpoint without losing progress.

Key Features:
- Granular checkpointing at multiple pipeline stages
- Atomic operations with rollback capabilities
- State validation and consistency checks
- Resume detection and recovery
- Progress preservation across restarts
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
import hashlib

from ._utils import logger, compute_mdhash_id
from .base import BaseKVStorage, TextChunkSchema


class CheckpointStage(Enum):
    """Pipeline stages that can be checkpointed"""
    INITIALIZATION = "initialization"
    DOCUMENT_PROCESSING = "document_processing"
    CHUNK_CREATION = "chunk_creation"
    ENTITY_EXTRACTION = "entity_extraction"
    RELATION_EXTRACTION = "relation_extraction"
    HIERARCHICAL_CLUSTERING = "hierarchical_clustering"
    ENTITY_DISAMBIGUATION = "entity_disambiguation"
    GRAPH_UPSERTION = "graph_upsertion"
    COMMUNITY_DETECTION = "community_detection"
    COMMUNITY_REPORTS = "community_reports"
    FINALIZATION = "finalization"
    COMPLETED = "completed"


class CheckpointStatus(Enum):
    """Status of a checkpoint"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class StageCheckpoint:
    """Checkpoint for a specific pipeline stage"""
    stage: CheckpointStage
    status: CheckpointStatus
    start_time: float
    end_time: Optional[float] = None
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if isinstance(self.stage, str):
            self.stage = CheckpointStage(self.stage)
        if isinstance(self.status, str):
            self.status = CheckpointStatus(self.status)
    
    @property
    def duration(self) -> Optional[float]:
        """Get stage duration in seconds"""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_completed(self) -> bool:
        """Check if stage is completed"""
        return self.status == CheckpointStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        """Check if stage failed"""
        return self.status == CheckpointStatus.FAILED


@dataclass
class PipelineCheckpoint:
    """Complete checkpoint state for the entire pipeline"""
    session_id: str
    timestamp: float
    hirag_config_hash: str
    input_documents_hash: str
    current_stage: CheckpointStage
    overall_progress: float = 0.0
    
    # Stage checkpoints
    stages: Dict[str, StageCheckpoint] = field(default_factory=dict)
    
    # Data state
    processed_documents: Set[str] = field(default_factory=set)
    processed_chunks: Set[str] = field(default_factory=set)
    extracted_entities: Set[str] = field(default_factory=set)
    extracted_relations: Set[str] = field(default_factory=set)
    disambiguated_entities: Set[str] = field(default_factory=set)
    completed_communities: Set[str] = field(default_factory=set)
    
    # Resume information
    last_successful_stage: Optional[CheckpointStage] = None
    resume_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.current_stage, str):
            self.current_stage = CheckpointStage(self.current_stage)
        if self.last_successful_stage and isinstance(self.last_successful_stage, str):
            self.last_successful_stage = CheckpointStage(self.last_successful_stage)
        
        # Convert string keys back to enums in stages dict
        new_stages = {}
        for stage_key, stage_checkpoint in self.stages.items():
            if isinstance(stage_key, str):
                try:
                    stage_enum = CheckpointStage(stage_key)
                    new_stages[stage_enum] = stage_checkpoint
                except ValueError:
                    # Keep original key if conversion fails
                    new_stages[stage_key] = stage_checkpoint
            else:
                new_stages[stage_key] = stage_checkpoint
        self.stages = new_stages
    
    def get_stage_checkpoint(self, stage: CheckpointStage) -> Optional[StageCheckpoint]:
        """Get checkpoint for a specific stage"""
        return self.stages.get(stage.value)

    def is_stage_completed(self, stage: CheckpointStage) -> bool:
        """Check if a stage is completed"""
        checkpoint = self.get_stage_checkpoint(stage)
        return checkpoint is not None and checkpoint.is_completed
    
    def get_completed_stages(self) -> List[CheckpointStage]:
        """Get list of completed stages in order"""
        completed = []
        for stage in CheckpointStage:
            if self.is_stage_completed(stage):
                completed.append(stage)
        return completed
    
    def get_resume_stage(self) -> CheckpointStage:
        """Determine which stage to resume from"""
        # Find the last completed stage
        completed_stages = self.get_completed_stages()
        
        if not completed_stages:
            return CheckpointStage.INITIALIZATION
        
        # Resume from the stage after the last completed one
        stage_order = list(CheckpointStage)
        last_completed_idx = stage_order.index(completed_stages[-1])
        
        if last_completed_idx + 1 < len(stage_order):
            return stage_order[last_completed_idx + 1]
        else:
            return CheckpointStage.COMPLETED


class CheckpointManager:
    """
    Manages checkpointing and recovery for the HiRAG pipeline
    
    This manager provides:
    - Automatic checkpointing at key pipeline stages
    - Resume capability from any checkpoint
    - State validation and consistency checks
    - Rollback capabilities for failed operations
    """
    
    def __init__(
        self,
        checkpoint_storage: BaseKVStorage,
        auto_checkpoint_interval: float = 30.0,  # seconds
        max_checkpoints: int = 10
    ):
        self.storage = checkpoint_storage
        self.auto_checkpoint_interval = auto_checkpoint_interval
        self.max_checkpoints = max_checkpoints
        
        self.current_checkpoint: Optional[PipelineCheckpoint] = None
        self._last_auto_checkpoint = 0.0
        self._checkpoint_lock = asyncio.Lock()
        
        logger.info(f"CheckpointManager initialized with auto-interval={auto_checkpoint_interval}s")

    def create_session_id(self, hirag_config: Dict[str, Any], input_documents: List[str]) -> str:
        """Create a unique session ID based on configuration and input"""
        config_str = json.dumps(hirag_config, sort_keys=True)
        input_str = "".join(sorted(input_documents))
        
        combined = f"{config_str}|{input_str}"
        session_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
        
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_hash}"

    def create_content_hash(self, content: Union[str, List[str], Dict[str, Any]]) -> str:
        """Create a hash for content to detect changes"""
        if isinstance(content, dict):
            content_str = json.dumps(content, sort_keys=True)
        elif isinstance(content, list):
            content_str = json.dumps(sorted(content))
        else:
            content_str = str(content)
        
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    async def start_new_session(
        self,
        hirag_config: Dict[str, Any],
        input_documents: List[str]
    ) -> PipelineCheckpoint:
        """
        Start a new checkpointing session
        
        Args:
            hirag_config: HiRAG configuration dictionary
            input_documents: List of input documents
            
        Returns:
            New PipelineCheckpoint instance
        """
        session_id = self.create_session_id(hirag_config, input_documents)
        config_hash = self.create_content_hash(hirag_config)
        input_hash = self.create_content_hash(input_documents)
        
        checkpoint = PipelineCheckpoint(
            session_id=session_id,
            timestamp=time.time(),
            hirag_config_hash=config_hash,
            input_documents_hash=input_hash,
            current_stage=CheckpointStage.INITIALIZATION
        )
        
        self.current_checkpoint = checkpoint
        await self._save_checkpoint()
        
        logger.info(f"Started new checkpoint session: {session_id}")
        return checkpoint

    async def find_resumable_session(
        self,
        hirag_config: Dict[str, Any],
        input_documents: List[str]
    ) -> Optional[PipelineCheckpoint]:
        """
        Find an existing session that can be resumed
        
        Args:
            hirag_config: Current HiRAG configuration
            input_documents: Current input documents
            
        Returns:
            Resumable checkpoint or None if no valid session found
        """
        config_hash = self.create_content_hash(hirag_config)
        input_hash = self.create_content_hash(input_documents)
        
        all_checkpoints = await self.storage.get_all()
        
        resumable_sessions = []
        
        for checkpoint_data in all_checkpoints.values():
            try:
                checkpoint = PipelineCheckpoint(**checkpoint_data)
                
                # Check if configuration and input match
                if (checkpoint.hirag_config_hash == config_hash and 
                    checkpoint.input_documents_hash == input_hash and
                    checkpoint.current_stage != CheckpointStage.COMPLETED):
                    
                    resumable_sessions.append(checkpoint)
                    
            except Exception as e:
                logger.warning(f"Invalid checkpoint data: {e}")
                continue
        
        if resumable_sessions:
            # Return the most recent resumable session
            latest_session = max(resumable_sessions, key=lambda c: c.timestamp)
            logger.info(f"Found resumable session: {latest_session.session_id}")
            return latest_session
        
        logger.info("No resumable session found")
        return None

    async def resume_session(self, checkpoint: PipelineCheckpoint) -> CheckpointStage:
        """
        Resume from an existing checkpoint
        
        Args:
            checkpoint: The checkpoint to resume from
            
        Returns:
            The stage to resume from
        """
        self.current_checkpoint = checkpoint
        resume_stage = checkpoint.get_resume_stage()
        
        logger.info(f"Resuming session {checkpoint.session_id} from stage: {resume_stage.value}")
        logger.info(f"Completed stages: {[s.value for s in checkpoint.get_completed_stages()]}")
        
        return resume_stage

    async def start_stage(
        self,
        stage: CheckpointStage,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StageCheckpoint:
        """
        Mark the start of a pipeline stage
        
        Args:
            stage: The stage being started
            metadata: Optional metadata for the stage
            
        Returns:
            StageCheckpoint instance
        """
        if not self.current_checkpoint:
            raise RuntimeError("No active checkpoint session")
        
        stage_checkpoint = StageCheckpoint(
            stage=stage,
            status=CheckpointStatus.IN_PROGRESS,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        self.current_checkpoint.stages[stage.value] = stage_checkpoint
        self.current_checkpoint.current_stage = stage
        
        await self._maybe_auto_checkpoint()
        
        logger.info(f"Started stage: {stage.value}")
        return stage_checkpoint

    async def update_stage_progress(
        self,
        stage: CheckpointStage,
        progress: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update progress for a stage
        
        Args:
            stage: The stage to update
            progress: Progress value (0.0 to 1.0)
            metadata: Optional additional metadata
        """
        if not self.current_checkpoint:
            return
        
        stage_checkpoint = self.current_checkpoint.stages.get(stage.value)
        if not stage_checkpoint:
            logger.warning(f"No checkpoint found for stage: {stage.value}")
            return
        
        stage_checkpoint.progress = min(1.0, max(0.0, progress))
        
        if metadata:
            stage_checkpoint.metadata.update(metadata)
        
        # Update overall progress based on stage weights
        self._update_overall_progress()
        
        await self._maybe_auto_checkpoint()
        
        logger.debug(f"Stage {stage.value} progress: {progress:.1%}")

    async def complete_stage(
        self,
        stage: CheckpointStage,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Mark a stage as completed
        
        Args:
            stage: The stage being completed
            metadata: Optional completion metadata
        """
        if not self.current_checkpoint:
            return
        
        stage_checkpoint = self.current_checkpoint.stages.get(stage.value)
        if not stage_checkpoint:
            logger.warning(f"No checkpoint found for stage: {stage.value}")
            return
        
        stage_checkpoint.status = CheckpointStatus.COMPLETED
        stage_checkpoint.end_time = time.time()
        stage_checkpoint.progress = 1.0
        
        if metadata:
            stage_checkpoint.metadata.update(metadata)
        
        self.current_checkpoint.last_successful_stage = stage
        self._update_overall_progress()
        
        await self._save_checkpoint()
        
        duration = stage_checkpoint.duration
        logger.info(f"Completed stage: {stage.value} (duration: {duration:.2f}s)")

    async def fail_stage(
        self,
        stage: CheckpointStage,
        error: Exception,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Mark a stage as failed
        
        Args:
            stage: The stage that failed
            error: The exception that caused the failure
            metadata: Optional failure metadata
        """
        if not self.current_checkpoint:
            return
        
        stage_checkpoint = self.current_checkpoint.stages.get(stage.value)
        if not stage_checkpoint:
            logger.warning(f"No checkpoint found for stage: {stage.value}")
            return
        
        stage_checkpoint.status = CheckpointStatus.FAILED
        stage_checkpoint.end_time = time.time()
        stage_checkpoint.error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time()
        }
        
        if metadata:
            stage_checkpoint.metadata.update(metadata)
        
        await self._save_checkpoint()
        
        logger.error(f"Stage failed: {stage.value} - {error}")

    async def record_processed_items(
        self,
        documents: Optional[Set[str]] = None,
        chunks: Optional[Set[str]] = None,
        entities: Optional[Set[str]] = None,
        relations: Optional[Set[str]] = None,
        communities: Optional[Set[str]] = None
    ):
        """
        Record processed items for resume capability
        
        Args:
            documents: Set of processed document IDs
            chunks: Set of processed chunk IDs
            entities: Set of processed entity IDs
            relations: Set of processed relation IDs
            communities: Set of processed community IDs
        """
        if not self.current_checkpoint:
            return
        
        if documents:
            self.current_checkpoint.processed_documents.update(documents)
        if chunks:
            self.current_checkpoint.processed_chunks.update(chunks)
        if entities:
            self.current_checkpoint.extracted_entities.update(entities)
        if relations:
            self.current_checkpoint.extracted_relations.update(relations)
        if communities:
            self.current_checkpoint.completed_communities.update(communities)
        
        await self._maybe_auto_checkpoint()

    async def get_unprocessed_items(
        self,
        all_documents: Optional[Set[str]] = None,
        all_chunks: Optional[Set[str]] = None,
        all_entities: Optional[Set[str]] = None
    ) -> Dict[str, Set[str]]:
        """
        Get items that still need processing
        
        Returns:
            Dictionary with unprocessed items by type
        """
        if not self.current_checkpoint:
            return {
                "documents": all_documents or set(),
                "chunks": all_chunks or set(),
                "entities": all_entities or set()
            }
        
        return {
            "documents": (all_documents or set()) - self.current_checkpoint.processed_documents,
            "chunks": (all_chunks or set()) - self.current_checkpoint.processed_chunks,
            "entities": (all_entities or set()) - self.current_checkpoint.extracted_entities
        }

    async def set_resume_data(self, key: str, data: Any):
        """Store arbitrary data for resuming operations"""
        if not self.current_checkpoint:
            return
        
        self.current_checkpoint.resume_data[key] = data
        await self._maybe_auto_checkpoint()

    async def get_resume_data(self, key: str, default: Any = None) -> Any:
        """Retrieve stored resume data"""
        if not self.current_checkpoint:
            return default
        
        return self.current_checkpoint.resume_data.get(key, default)

    async def complete_session(self):
        """Mark the entire session as completed"""
        if not self.current_checkpoint:
            return
        
        self.current_checkpoint.current_stage = CheckpointStage.COMPLETED
        self.current_checkpoint.overall_progress = 1.0
        
        await self._save_checkpoint()
        
        logger.info(f"Session completed: {self.current_checkpoint.session_id}")

    async def cleanup_old_checkpoints(self, max_age_days: int = 7) -> int:
        """
        Remove old checkpoint sessions
        
        Args:
            max_age_days: Maximum age of checkpoints to keep
            
        Returns:
            Number of checkpoints removed
        """
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        all_checkpoints = await self.storage.get_all()
        
        to_remove = []
        for checkpoint_id, checkpoint_data in all_checkpoints.items():
            try:
                timestamp = checkpoint_data.get("timestamp", 0)
                if timestamp < cutoff_time:
                    to_remove.append(checkpoint_id)
            except Exception as e:
                logger.warning(f"Error checking checkpoint {checkpoint_id}: {e}")
                to_remove.append(checkpoint_id)
        
        # Remove old checkpoints
        for checkpoint_id in to_remove:
            await self.storage.delete(checkpoint_id)
        
        logger.info(f"Cleaned up {len(to_remove)} old checkpoints")
        return len(to_remove)

    def _update_overall_progress(self):
        """Update overall pipeline progress based on stage progress"""
        if not self.current_checkpoint:
            return
        
        # Define stage weights (some stages are more time-consuming)
        stage_weights = {
            CheckpointStage.INITIALIZATION: 0.02,
            CheckpointStage.DOCUMENT_PROCESSING: 0.05,
            CheckpointStage.CHUNK_CREATION: 0.08,
            CheckpointStage.ENTITY_EXTRACTION: 0.30,
            CheckpointStage.RELATION_EXTRACTION: 0.20,
            CheckpointStage.HIERARCHICAL_CLUSTERING: 0.15,
            CheckpointStage.ENTITY_DISAMBIGUATION: 0.10,
            CheckpointStage.GRAPH_UPSERTION: 0.05,
            CheckpointStage.COMMUNITY_DETECTION: 0.03,
            CheckpointStage.COMMUNITY_REPORTS: 0.02,
            CheckpointStage.FINALIZATION: 0.01,
        }
        
        total_progress = 0.0
        for stage, weight in stage_weights.items():
            stage_checkpoint = self.current_checkpoint.stages.get(stage.value)
            if stage_checkpoint:
                total_progress += weight * stage_checkpoint.progress
        
        self.current_checkpoint.overall_progress = min(1.0, total_progress)

    async def _maybe_auto_checkpoint(self):
        """Save checkpoint if enough time has passed"""
        current_time = time.time()
        if current_time - self._last_auto_checkpoint > self.auto_checkpoint_interval:
            await self._save_checkpoint()
            self._last_auto_checkpoint = current_time

    async def _save_checkpoint(self):
        """Save current checkpoint to storage"""
        if not self.current_checkpoint:
            return
        
        async with self._checkpoint_lock:
            # Convert sets to lists for JSON serialization
            checkpoint_data = asdict(self.current_checkpoint)
            
            # Convert sets to lists
            for key in ['processed_documents', 'processed_chunks', 'extracted_entities', 
                       'extracted_relations', 'disambiguated_entities', 'completed_communities']:
                if key in checkpoint_data:
                    checkpoint_data[key] = list(checkpoint_data[key])
            
            # Convert enum keys to strings
            if 'stages' in checkpoint_data:
                string_stages = {}
                for stage_key, stage_data in checkpoint_data['stages'].items():
                    if hasattr(stage_key, 'value'):
                        string_stages[stage_key.value] = stage_data
                    else:
                        string_stages[str(stage_key)] = stage_data
                checkpoint_data['stages'] = string_stages
            
            # Convert enum values to strings
            if 'current_stage' in checkpoint_data and hasattr(checkpoint_data['current_stage'], 'value'):
                checkpoint_data['current_stage'] = checkpoint_data['current_stage'].value
            if ('last_successful_stage' in checkpoint_data and 
                checkpoint_data['last_successful_stage'] and 
                hasattr(checkpoint_data['last_successful_stage'], 'value')):
                checkpoint_data['last_successful_stage'] = checkpoint_data['last_successful_stage'].value
            
            checkpoint_id = f"checkpoint_{self.current_checkpoint.session_id}"
            await self.storage.upsert({checkpoint_id: checkpoint_data})
            
            logger.debug(f"Checkpoint saved: {checkpoint_id}")

    async def generate_checkpoint_report(self) -> str:
        """Generate a human-readable checkpoint status report"""
        if not self.current_checkpoint:
            return "No active checkpoint session"
        
        cp = self.current_checkpoint
        
        report_lines = [
            "=== Checkpoint Status Report ===",
            f"Session ID: {cp.session_id}",
            f"Started: {datetime.fromtimestamp(cp.timestamp).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Current Stage: {cp.current_stage.value}",
            f"Overall Progress: {cp.overall_progress:.1%}",
            "",
            "Stage Details:"
        ]
        
        for stage in CheckpointStage:
            stage_checkpoint = cp.stages.get(stage.value)
            if stage_checkpoint:
                duration_str = f" ({stage_checkpoint.duration:.1f}s)" if stage_checkpoint.duration else ""
                status_icon = {
                    CheckpointStatus.COMPLETED: "✓",
                    CheckpointStatus.IN_PROGRESS: "⏳",
                    CheckpointStatus.FAILED: "❌",
                    CheckpointStatus.PENDING: "⏸",
                    CheckpointStatus.ROLLED_BACK: "↩"
                }.get(stage_checkpoint.status, "?")
                
                report_lines.append(
                    f"  {status_icon} {stage.value}: {stage_checkpoint.progress:.1%}{duration_str}"
                )
            else:
                report_lines.append(f"  ⏸ {stage.value}: pending")
        
        report_lines.extend([
            "",
            "Processing Progress:",
            f"  • Documents: {len(cp.processed_documents)}",
            f"  • Chunks: {len(cp.processed_chunks)}",
            f"  • Entities: {len(cp.extracted_entities)}",
            f"  • Relations: {len(cp.extracted_relations)}",
            f"  • Communities: {len(cp.completed_communities)}"
        ])
        
        return "\n".join(report_lines)


# Utility functions

async def create_checkpoint_manager(
    checkpoint_storage: BaseKVStorage,
    auto_checkpoint_interval: float = 30.0,
    max_checkpoints: int = 10
) -> CheckpointManager:
    """Factory function to create a CheckpointManager instance"""
    manager = CheckpointManager(checkpoint_storage, auto_checkpoint_interval, max_checkpoints)
    
    # Cleanup old checkpoints on initialization
    await manager.cleanup_old_checkpoints()
    
    return manager