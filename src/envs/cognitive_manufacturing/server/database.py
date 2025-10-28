"""Database manager for persistent storage of production data."""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Any
from dataclasses import asdict

try:
    from sqlalchemy import (
        create_engine,
        Column,
        String,
        Float,
        Integer,
        Boolean,
        DateTime,
        ForeignKey,
        Text,
        JSON,
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship
    from sqlalchemy.dialects.postgresql import UUID, JSONB
    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    # Provide dummy classes for type hints when SQLAlchemy not installed
    Base = type('Base', (), {})  # type: ignore
    Column = None  # type: ignore
    String = None  # type: ignore
    Float = None  # type: ignore
    Integer = None  # type: ignore
    Boolean = None  # type: ignore
    DateTime = None  # type: ignore
    ForeignKey = None  # type: ignore
    Text = None  # type: ignore
    JSON = None  # type: ignore
    UUID = None  # type: ignore
    JSONB = None  # type: ignore
    create_engine = None  # type: ignore
    sessionmaker = None  # type: ignore
    relationship = None  # type: ignore


class ProductionRun(Base):
    """Production run record."""

    __tablename__ = "production_runs"

    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_name = Column(String(255), nullable=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    mode = Column(String(20), nullable=False)  # 'single' or 'multi_machine'
    total_steps = Column(Integer, default=0)
    simulation_time = Column(Float, default=0.0)
    cumulative_reward = Column(Float, default=0.0)
    status = Column(String(20), default="in_progress")  # 'in_progress', 'completed', 'failed'
    run_metadata = Column(JSON, nullable=True)

    # Relationships
    sensor_readings = relationship("SensorReading", back_populates="run", cascade="all, delete-orphan")
    machine_events = relationship("MachineEvent", back_populates="run", cascade="all, delete-orphan")
    production_units = relationship("ProductionUnit", back_populates="run", cascade="all, delete-orphan")


class SensorReading(Base):
    """Time-series sensor data."""

    __tablename__ = "sensor_readings"

    reading_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("production_runs.run_id"), nullable=False)
    machine_id = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    simulation_time = Column(Float, nullable=False)

    # Sensor values
    temperature = Column(Float)
    vibration = Column(Float)
    speed = Column(Float)
    health_score = Column(Float)
    wear_level = Column(Float)
    production_output = Column(Float)
    status = Column(String(20))

    # Relationship
    run = relationship("ProductionRun", back_populates="sensor_readings")


class MachineEvent(Base):
    """Machine events (speed changes, maintenance, alerts, etc.)."""

    __tablename__ = "machine_events"

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("production_runs.run_id"), nullable=False)
    machine_id = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    simulation_time = Column(Float, nullable=False)
    event_type = Column(String(50), nullable=False)  # 'speed_change', 'maintenance', 'alert', etc.
    data = Column(JSON, nullable=True)

    # Relationship
    run = relationship("ProductionRun", back_populates="machine_events")


class ProductionUnit(Base):
    """Production unit records (Phase 1)."""

    __tablename__ = "production_units"

    unit_id = Column(String(50), primary_key=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("production_runs.run_id"), nullable=False)
    created_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    stage = Column(String(20), nullable=False)  # 'prep', 'assembly', 'finishing', 'qc'
    quality = Column(Float, default=1.0)
    quality_score = Column(Float, default=1.0)  # Alias for quality
    passed_qc = Column(Boolean, nullable=True)
    machine_id = Column(String(10), nullable=True)  # Machine that produced this unit
    unit_metadata = Column(JSON, nullable=True)  # Production parameters (speed, temp, etc.)

    # Relationship
    run = relationship("ProductionRun", back_populates="production_units")


class KnowledgeBase(Base):
    """Knowledge base documents with vector embeddings."""

    __tablename__ = "knowledge_base"

    doc_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    doc_type = Column(String(50), nullable=False)  # 'maintenance', 'troubleshooting', 'safety', etc.
    doc_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Vector embedding stored as JSON array (pgvector would use VECTOR type)
    embedding = Column(JSON, nullable=True)


class DatabaseManager:
    """Manages PostgreSQL connections and operations."""

    def __init__(self, connection_string: str):
        """Initialize database connection.

        Args:
            connection_string: PostgreSQL connection string
                Example: "postgresql://user:pass@localhost/dbname"
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is required for database functionality. "
                "Install with: pip install sqlalchemy psycopg2-binary"
            )

        self.engine = create_engine(connection_string, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self):
        """Create all database tables if they don't exist."""
        Base.metadata.create_all(self.engine)

    def drop_tables(self):
        """Drop all database tables (use with caution!)."""
        Base.metadata.drop_all(self.engine)

    # Production Run Operations

    def create_run(self, mode: str, run_name: str | None = None, metadata: dict | None = None) -> str:
        """Create a new production run.

        Args:
            mode: 'single' or 'multi_machine'
            run_name: Optional name for the run
            metadata: Optional metadata dict

        Returns:
            run_id as string
        """
        session = self.SessionLocal()
        try:
            run = ProductionRun(
                run_name=run_name,
                start_time=datetime.utcnow(),
                mode=mode,
                run_metadata=metadata,
            )
            session.add(run)
            session.commit()
            run_id = str(run.run_id)
            return run_id
        finally:
            session.close()

    def complete_run(self, run_id: str, total_steps: int, simulation_time: float, cumulative_reward: float):
        """Mark a production run as completed.

        Args:
            run_id: Run identifier
            total_steps: Total number of steps
            simulation_time: Total simulation time (hours)
            cumulative_reward: Total cumulative reward
        """
        session = self.SessionLocal()
        try:
            run = session.query(ProductionRun).filter_by(run_id=uuid.UUID(run_id)).first()
            if run:
                run.end_time = datetime.utcnow()
                run.total_steps = total_steps
                run.simulation_time = simulation_time
                run.cumulative_reward = cumulative_reward
                run.status = "completed"
                session.commit()
        finally:
            session.close()

    def save_sensor_reading(
        self,
        run_id: str,
        machine_id: str,
        simulation_time: float,
        temperature: float,
        vibration: float,
        speed: float,
        health_score: float,
        wear_level: float,
        production_output: float,
        status: str,
    ):
        """Save a sensor reading to the database.

        Args:
            run_id: Production run ID
            machine_id: Machine identifier
            simulation_time: Simulation time in hours
            temperature: Temperature reading
            vibration: Vibration reading
            speed: Speed reading
            health_score: Health score
            wear_level: Wear level
            production_output: Production output
            status: Machine status
        """
        session = self.SessionLocal()
        try:
            reading = SensorReading(
                run_id=uuid.UUID(run_id),
                machine_id=machine_id,
                timestamp=datetime.utcnow(),
                simulation_time=simulation_time,
                temperature=temperature,
                vibration=vibration,
                speed=speed,
                health_score=health_score,
                wear_level=wear_level,
                production_output=production_output,
                status=status,
            )
            session.add(reading)
            session.commit()
        finally:
            session.close()

    def save_machine_event(
        self,
        run_id: str,
        machine_id: str,
        simulation_time: float,
        event_type: str,
        data: dict,
    ):
        """Save a machine event to the database.

        Args:
            run_id: Production run ID
            machine_id: Machine identifier
            simulation_time: Simulation time in hours
            event_type: Type of event
            data: Event data dict
        """
        session = self.SessionLocal()
        try:
            event = MachineEvent(
                run_id=uuid.UUID(run_id),
                machine_id=machine_id,
                timestamp=datetime.utcnow(),
                simulation_time=simulation_time,
                event_type=event_type,
                data=data,
            )
            session.add(event)
            session.commit()
        finally:
            session.close()

    def query_runs(self, filters: dict | None = None, limit: int = 100) -> list[dict]:
        """Query production runs with optional filters.

        Args:
            filters: Optional filters dict (start_date, end_date, min_reward, status, mode)
            limit: Maximum number of results

        Returns:
            List of production run dicts
        """
        session = self.SessionLocal()
        try:
            query = session.query(ProductionRun)

            if filters:
                if "start_date" in filters:
                    query = query.filter(ProductionRun.start_time >= filters["start_date"])
                if "end_date" in filters:
                    query = query.filter(ProductionRun.start_time <= filters["end_date"])
                if "min_reward" in filters:
                    query = query.filter(ProductionRun.cumulative_reward >= filters["min_reward"])
                if "status" in filters:
                    query = query.filter(ProductionRun.status == filters["status"])
                if "mode" in filters:
                    query = query.filter(ProductionRun.mode == filters["mode"])

            runs = query.order_by(ProductionRun.start_time.desc()).limit(limit).all()

            return [
                {
                    "run_id": str(run.run_id),
                    "run_name": run.run_name,
                    "start_time": run.start_time.isoformat() if run.start_time else None,
                    "end_time": run.end_time.isoformat() if run.end_time else None,
                    "mode": run.mode,
                    "total_steps": run.total_steps,
                    "simulation_time": run.simulation_time,
                    "cumulative_reward": run.cumulative_reward,
                    "status": run.status,
                    "metadata": run.run_metadata,
                }
                for run in runs
            ]
        finally:
            session.close()

    def get_run_count(self) -> int:
        """Get total number of production runs.

        Returns:
            Number of runs in database
        """
        session = self.SessionLocal()
        try:
            return session.query(ProductionRun).count()
        finally:
            session.close()

    def execute_sql(self, query: str) -> list[dict]:
        """Execute a read-only SQL query.

        Args:
            query: SQL SELECT query

        Returns:
            List of result dicts

        Raises:
            ValueError: If query is not a SELECT statement
        """
        # Safety check: only allow SELECT
        query_upper = query.strip().upper()
        if not query_upper.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")

        # Block dangerous keywords
        dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                raise ValueError(f"Query contains forbidden keyword: {keyword}")

        session = self.SessionLocal()
        try:
            result = session.execute(query)
            columns = result.keys()
            rows = result.fetchall()

            return [dict(zip(columns, row)) for row in rows]
        finally:
            session.close()

    # Knowledge Base Operations

    def add_knowledge(
        self,
        title: str,
        content: str,
        doc_type: str,
        embedding: list[float] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Add a document to the knowledge base.

        Args:
            title: Document title
            content: Document content
            doc_type: Type of document
            embedding: Vector embedding (if available)
            metadata: Optional metadata

        Returns:
            doc_id as string
        """
        session = self.SessionLocal()
        try:
            doc = KnowledgeBase(
                title=title,
                content=content,
                doc_type=doc_type,
                embedding=embedding,
                doc_metadata=metadata,
            )
            session.add(doc)
            session.commit()
            doc_id = str(doc.doc_id)
            return doc_id
        finally:
            session.close()

    def search_knowledge(
        self,
        query_embedding: list[float],
        doc_type: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Search knowledge base using vector similarity.

        Args:
            query_embedding: Query vector embedding
            doc_type: Optional document type filter
            top_k: Number of results to return

        Returns:
            List of matching documents with similarity scores
        """
        session = self.SessionLocal()
        try:
            query = session.query(KnowledgeBase)

            if doc_type:
                query = query.filter(KnowledgeBase.doc_type == doc_type)

            docs = query.all()

            # Compute cosine similarity (simple implementation)
            def cosine_similarity(a: list[float], b: list[float]) -> float:
                if not a or not b:
                    return 0.0
                dot_product = sum(x * y for x, y in zip(a, b))
                mag_a = sum(x * x for x in a) ** 0.5
                mag_b = sum(x * x for x in b) ** 0.5
                if mag_a == 0 or mag_b == 0:
                    return 0.0
                return dot_product / (mag_a * mag_b)

            results = []
            for doc in docs:
                if doc.embedding:
                    similarity = cosine_similarity(query_embedding, doc.embedding)
                    results.append({
                        "doc_id": str(doc.doc_id),
                        "title": doc.title,
                        "content": doc.content,
                        "doc_type": doc.doc_type,
                        "similarity": similarity,
                        "metadata": doc.doc_metadata,
                    })

            # Sort by similarity and return top-k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
        finally:
            session.close()

    # =====================================================================
    # Phase 3: ML Training Data Retrieval
    # =====================================================================

    def get_sensor_readings(self, limit: int = 1000, machine_id: str | None = None) -> list[dict]:
        """Get sensor readings for ML model training.

        Args:
            limit: Maximum number of readings to return
            machine_id: Optional filter by machine ID

        Returns:
            List of sensor reading dicts
        """
        session = self.Session()
        try:
            query = session.query(SensorReading)

            if machine_id:
                query = query.filter(SensorReading.machine_id == machine_id)

            readings = query.order_by(SensorReading.timestamp.desc()).limit(limit).all()

            return [
                {
                    "machine_id": r.machine_id,
                    "temperature": r.temperature,
                    "vibration": r.vibration,
                    "speed": r.speed,
                    "health_score": r.health_score,
                    "wear_level": r.wear_level,
                    "production_output": r.production_output,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "failed": r.health_score < 30 if r.health_score else False,
                }
                for r in readings
            ]
        finally:
            session.close()

    def get_production_units(self, limit: int = 1000, machine_id: str | None = None) -> list[dict]:
        """Get production units for quality prediction training.

        Args:
            limit: Maximum number of units to return
            machine_id: Optional filter by machine ID

        Returns:
            List of production unit dicts
        """
        session = self.Session()
        try:
            query = session.query(ProductionUnit)

            if machine_id:
                query = query.filter(ProductionUnit.machine_id == machine_id)

            units = query.order_by(ProductionUnit.completed_at.desc()).limit(limit).all()

            return [
                {
                    "machine_id": u.machine_id,
                    "speed": u.unit_metadata.get("speed", 0) if u.unit_metadata else 0,
                    "temperature": u.unit_metadata.get("temperature", 0) if u.unit_metadata else 0,
                    "vibration": u.unit_metadata.get("vibration", 0) if u.unit_metadata else 0,
                    "wear_level": u.unit_metadata.get("wear_level", 0) if u.unit_metadata else 0,
                    "quality": u.quality_score,
                }
                for u in units
            ]
        finally:
            session.close()
