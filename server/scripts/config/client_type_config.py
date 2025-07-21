"""
Client Type Configuration
========================

Configuration for specific client type scoring with complete isolation.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict

from ..enums.client_type import ClientType
from .component_config import ComponentConfig

logger = logging.getLogger(__name__)


@dataclass
class ClientTypeConfig:
    """Configuration for specific client type scoring with complete isolation."""
    client_type: ClientType
    components: Dict[str, ComponentConfig] = field(default_factory=dict)
    description: str = ""
    
    def add_component(self, component_config: ComponentConfig) -> None:
        """Add component with validation."""
        self.components[component_config.name] = component_config
        
        # Validate total component contributions
        total_contribution = sum(c.max_contribution_pct for c in self.components.values())
        if total_contribution > 1.0:
            logger.warning(f"Client type {self.client_type.value}: total contributions exceed 100% ({total_contribution:.1%})")
    
    def get_total_max_contribution(self) -> float:
        """Get total maximum contribution across all components."""
        return sum(c.max_contribution_pct for c in self.components.values())