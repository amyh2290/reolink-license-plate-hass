"""Sensor platform for Reolink License Plate Detection."""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the sensor platform."""
    coordinator = hass.data[DOMAIN][config_entry.entry_id]
    
    entities = []
    
    # Create sensors for each camera
    for camera_name in coordinator.cameras.keys():
        entities.extend([
            LicensePlateSensor(coordinator, camera_name, "last_detection"),
            LicensePlateSensor(coordinator, camera_name, "detection_count"),
            LicensePlateSensor(coordinator, camera_name, "last_allowed_plate"),
            LicensePlateSensor(coordinator, camera_name, "camera_status"),
        ])
    
    # Global sensors
    entities.extend([
        GlobalLicensePlateSensor(coordinator, "total_detections"),
        GlobalLicensePlateSensor(coordinator, "allowed_plates_count"),
        GlobalLicensePlateSensor(coordinator, "garage_triggers_count"),
    ])
    
    async_add_entities(entities)


class LicensePlateSensor(SensorEntity):
    """Sensor for individual camera license plate detection."""

    def __init__(self, coordinator, camera_name: str, sensor_type: str):
        """Initialize the sensor."""
        self._coordinator = coordinator
        self._camera_name = camera_name
        self._sensor_type = sensor_type
        self._attr_unique_id = f"{DOMAIN}_{camera_name}_{sensor_type}"
        self._attr_name = f"{camera_name.title()} {sensor_type.replace('_', ' ').title()}"
        
        # Set up device info
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, camera_name)},
            name=f"Reolink Camera {camera_name.title()}",
            manufacturer="Reolink",
            model="License Plate Detector",
            sw_version="1.0.0",
        )
        
        # Initialize attributes
        self._attr_extra_state_attributes = {}
        self._detection_count = 0
        self._last_detection = None
        self._last_allowed_plate = None
        self._garage_triggers = 0

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return self._camera_name in self._coordinator.cameras

    @property
    def state(self) -> Any:
        """Return the state of the sensor."""
        if self._sensor_type == "last_detection":
            return self._last_detection or STATE_UNKNOWN
        elif self._sensor_type == "detection_count":
            return self._detection_count
        elif self._sensor_type == "last_allowed_plate":
            return self._last_allowed_plate or STATE_UNKNOWN
        elif self._sensor_type == "camera_status":
            return "Connected" if self.available else "Disconnected"
        
        return STATE_UNKNOWN

    @property
    def unit_of_measurement(self) -> Optional[str]:
        """Return the unit of measurement."""
        if self._sensor_type in ["detection_count"]:
            return "detections"
        return None

    @property
    def icon(self) -> str:
        """Return the icon."""
        if self._sensor_type == "last_detection":
            return "mdi:license"
        elif self._sensor_type == "detection_count":
            return "mdi:counter"
        elif self._sensor_type == "last_allowed_plate":
            return "mdi:check-circle"
        elif self._sensor_type == "camera_status":
            return "mdi:camera" if self.available else "mdi:camera-off"
        
        return "mdi:car"

    async def async_update(self) -> None:
        """Update the sensor state."""
        # Listen for detection events
        if hasattr(self._coordinator, 'last_detections'):
            camera_detections = [
                detection for detection in self._coordinator.last_detections
                if detection.get('camera') == self._camera_name
            ]
            
            if camera_detections:
                latest = max(camera_detections, key=lambda x: x.get('timestamp', ''))
                
                if self._sensor_type == "last_detection":
                    self._last_detection = latest.get('plate')
                    self._attr_extra_state_attributes = {
                        "timestamp": latest.get('timestamp'),
                        "allowed": latest.get('allowed'),
                        "confidence": latest.get('confidence', 0),
                    }
                
                elif self._sensor_type == "detection_count":
                    self._detection_count = len(camera_detections)
                    self._attr_extra_state_attributes = {
                        "last_24h": len([
                            d for d in camera_detections
                            if self._is_within_24h(d.get('timestamp'))
                        ]),
                        "allowed_detections": len([
                            d for d in camera_detections
                            if d.get('allowed')
                        ]),
                    }
                
                elif self._sensor_type == "last_allowed_plate":
                    allowed_detections = [d for d in camera_detections if d.get('allowed')]
                    if allowed_detections:
                        latest_allowed = max(allowed_detections, key=lambda x: x.get('timestamp', ''))
                        self._last_allowed_plate = latest_allowed.get('plate')
                        self._attr_extra_state_attributes = {
                            "timestamp": latest_allowed.get('timestamp'),
                            "garage_triggered": latest_allowed.get('garage_triggered', False),
                        }

    def _is_within_24h(self, timestamp_str: str) -> bool:
        """Check if timestamp is within the last 24 hours."""
        try:
            timestamp = dt_util.parse_datetime(timestamp_str)
            if timestamp:
                return dt_util.utcnow() - timestamp < timedelta(hours=24)
        except Exception:
            pass
        return False


class GlobalLicensePlateSensor(SensorEntity):
    """Global sensor for license plate detection system."""

    def __init__(self, coordinator, sensor_type: str):
        """Initialize the global sensor."""
        self._coordinator = coordinator
        self._sensor_type = sensor_type
        self._attr_unique_id = f"{DOMAIN}_global_{sensor_type}"
        self._attr_name = f"License Plate {sensor_type.replace('_', ' ').title()}"
        
        # Set up device info
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, "global")},
            name="Reolink License Plate System",
            manufacturer="Reolink",
            model="License Plate Detection System",
            sw_version="1.0.0",
        )

    @property
    def state(self) -> Any:
        """Return the state of the sensor."""
        if self._sensor_type == "total_detections":
            return len(getattr(self._coordinator, 'all_detections', []))
        elif self._sensor_type == "allowed_plates_count":
            return len(self._coordinator.allowed_plates)
        elif self._sensor_type == "garage_triggers_count":
            return getattr(self._coordinator, 'garage_trigger_count', 0)
        
        return 0

    @property
    def unit_of_measurement(self) -> Optional[str]:
        """Return the unit of measurement."""
        if self._sensor_type.endswith("_count") or self._sensor_type == "total_detections":
            return "items"
        return None

    @property
    def icon(self) -> str:
        """Return the icon."""
        if self._sensor_type == "total_detections":
            return "mdi:counter"
        elif self._sensor_type == "allowed_plates_count":
            return "mdi:format-list-bulleted"
        elif self._sensor_type == "garage_triggers_count":
            return "mdi:garage-open"
        
        return "mdi:car"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return additional state attributes."""
        if self._sensor_type == "total_detections":
            return {
                "cameras": list(self._coordinator.cameras.keys()),
                "detection_rate_per_hour": self._calculate_detection_rate(),
            }
        elif self._sensor_type == "allowed_plates_count":
            return {
                "allowed_plates": list(self._coordinator.allowed_plates),
                "last_updated": dt_util.utcnow().isoformat(),
            }
        elif self._sensor_type == "garage_triggers_count":
            return {
                "last_trigger": getattr(self._coordinator, 'last_garage_trigger', None),
                "auto_garage_enabled": any(
                    camera.get('config', {}).get('enable_auto_garage', False)
                    for camera in self._coordinator.cameras.values()
                ),
            }
        
        return {}

    def _calculate_detection_rate(self) -> float:
        """Calculate detections per hour."""
        try:
            all_detections = getattr(self._coordinator, 'all_detections', [])
            recent_detections = [
                d for d in all_detections
                if self._is_within_24h(d.get('timestamp', ''))
            ]
            return len(recent_detections) / 24  # Average per hour over 24h
        except Exception:
            return 0.0

    def _is_within_24h(self, timestamp_str: str) -> bool:
        """Check if timestamp is within the last 24 hours."""
        try:
            timestamp = dt_util.parse_datetime(timestamp_str)
            if timestamp:
                return dt_util.utcnow() - timestamp < timedelta(hours=24)
        except Exception:
            pass
        return False