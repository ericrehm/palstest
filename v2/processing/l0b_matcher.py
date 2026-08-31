"""
L0b Processing: Match PM data to shot pulses and produce L0b product.
"""

from pathlib import Path
from typing import List, Optional, Tuple
from ..data_types import L0Data, L0bData


class L0bMatcher:
    """
    Wrapper around match_power_data.py to enrich L0 with PM data.
    Produces L0b with provenance tracking.
    """
    
    def __init__(self, pm_data_dir: Optional[Path] = None, fallback_energy: float = 1.0):
        """
        Args:
            pm_data_dir: Directory containing PM (Power Monitor) files
            fallback_energy: Constant energy value if no PM match found
        """
        self.pm_data_dir = pm_data_dir
        self.fallback_energy = fallback_energy
        
        # Import match_power_data here to avoid circular dependencies
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from match_power_data import match_laser_to_pm
            self.matcher = match_laser_to_pm
        except ImportError as e:
            print(f"Warning: Could not import match_power_data: {e}")
            self.matcher = None
    
    def process(self, l0: L0Data) -> L0bData:
        """
        Enrich L0 with PM data.
        
        Args:
            l0: Raw L0 data
            
        Returns:
            L0bData with matched PM or fallback constant
        """
        
        # Attempt to match PM data if available
        if self.matcher is not None and self.pm_data_dir is not None:
            try:
                # Call existing match_power_data logic here
                power_a, power_b = self._match_pm(l0)
                match_quality = 0.95  # Placeholder
                time_offset_ns = 0.5  # Placeholder
                distance = 1  # Placeholder
                pm_source = "matched"
                pm_file_source = "TBD"  # Record which PM file
            except Exception as e:
                print(f"PM matching failed for {l0.timestamp_utc}: {e}")
                power_a, power_b = self.fallback_energy, self.fallback_energy / 10
                match_quality = None
                time_offset_ns = None
                distance = None
                pm_source = "constant"
                pm_file_source = None
        else:
            # Fallback to constant
            power_a, power_b = self.fallback_energy, self.fallback_energy / 10
            match_quality = None
            time_offset_ns = None
            distance = None
            pm_source = "constant"
            pm_file_source = None
        
        return L0bData(
            l0=l0,
            power_a_sum=power_a,
            power_b_sum=power_b,
            pm_source=pm_source,
            match_quality=match_quality,
            time_offset_ns=time_offset_ns,
            distance_matched_samples=distance,
            pm_file_source=pm_file_source,
            fallback_reason="No PM data" if pm_source == "constant" else None,
        )
    
    def _match_pm(self, l0: L0Data) -> Tuple[float, float]:
        """
        Internal: call match_power_data logic to get Power A/B for this shot.
        Placeholder implementation; details TBD based on integration.
        """
        # This will invoke the actual match_power_data.py logic
        # For now, return dummy values
        return 42.5, 3.2
    
    def find_pm_file(self, shot_file: Path) -> Optional[Path]:
        """
        Find corresponding PM file in same directory.
        Pattern: shot file 'D2_1 (D)_1.csv' -> PM file 'D2_1 (D)_PM_1.csv'
        (i.e., insert _PM before the final sequence number)
        """
        shot_dir = shot_file.parent
        shot_name = shot_file.stem  # filename without extension
        shot_ext = shot_file.suffix  # .csv
        
        # Find the last underscore-delimited segment (sequence number)
        # Pattern: base_sequence -> base_PM_sequence
        parts = shot_name.rsplit('_', 1)
        if len(parts) == 2:
            base, sequence = parts
            pm_name = f"{base}_PM_{sequence}{shot_ext}"
            pm_file = shot_dir / pm_name
            
            if pm_file.exists():
                return pm_file
        
        return None
