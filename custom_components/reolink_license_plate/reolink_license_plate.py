"""
Reolink License Plate Detection Integration for Home Assistant
"""
import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import cv2
import numpy as np
import voluptuous as vol
from homeassistant.components import persistent_notification
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_HOST,
    CONF_PASSWORD,
    CONF_PORT,
    CONF_USERNAME,
    Platform,
)
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

DOMAIN = "reolink_license_plate"
PLATFORMS = [Platform.SENSOR, Platform.SWITCH, Platform.CAMERA]

# Configuration constants
CONF_CAMERAS = "cameras"
CONF_CAMERA_NAME = "camera_name"
CONF_DETECTION_AREA = "detection_area"
CONF_ALLOWED_PLATES = "allowed_plates"
CONF_GARAGE_DOOR_ENTITY = "garage_door_entity"
CONF_DETECTION_SENSITIVITY = "detection_sensitivity"
CONF_MIN_CONFIDENCE = "min_confidence"
CONF_COOLDOWN_SECONDS = "cooldown_seconds"
CONF_ENABLE_NOTIFICATIONS = "enable_notifications"
CONF_ENABLE_AUTO_GARAGE = "enable_auto_garage"
CONF_LOG_ALL_DETECTIONS = "log_all_detections"

DEFAULT_PORT = 80
DEFAULT_SENSITIVITY = 0.7
DEFAULT_CONFIDENCE = 0.8
DEFAULT_COOLDOWN = 30

# Service schemas
SERVICE_ADD_PLATE = "add_allowed_plate"
SERVICE_REMOVE_PLATE = "remove_allowed_plate"
SERVICE_TRIGGER_DETECTION = "trigger_detection"

ADD_PLATE_SCHEMA = vol.Schema({
    vol.Required("plate_number"): cv.string,
    vol.Optional("description"): cv.string,
})

REMOVE_PLATE_SCHEMA = vol.Schema({
    vol.Required("plate_number"): cv.string,
})

TRIGGER_DETECTION_SCHEMA = vol.Schema({
    vol.Required("camera_name"): cv.string,
})


class ReolinkLicensePlateAPI:
    """API client for Reolink cameras with license plate detection."""
    
    def __init__(self, host: str, port: int, username: str, password: str):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.token = None
        self.session = None
        
    async def authenticate(self, session: aiohttp.ClientSession) -> bool:
        """Authenticate with the Reolink camera."""
        self.session = session
        
        auth_url = f"http://{self.host}:{self.port}/api.cgi?cmd=Login"
        auth_data = {
            "cmd": "Login",
            "param": {
                "User": {
                    "userName": self.username,
                    "password": self.password
                }
            }
        }
        
        try:
            async with session.post(auth_url, json=[auth_data]) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data[0]["code"] == 0:
                        self.token = data[0]["value"]["Token"]["name"]
                        return True
        except Exception as e:
            _LOGGER.error(f"Authentication failed: {e}")
            return False
        
        return False
    
    async def get_snapshot(self) -> Optional[bytes]:
        """Get a snapshot from the camera."""
        if not self.token or not self.session:
            return None
            
        snapshot_url = f"http://{self.host}:{self.port}/cgi-bin/api.cgi"
        params = {
            "cmd": "Snap",
            "channel": 0,
            "rs": dt_util.utcnow().strftime("%Y%m%d%H%M%S"),
            "token": self.token
        }
        
        try:
            async with self.session.get(snapshot_url, params=params) as resp:
                if resp.status == 200:
                    return await resp.read()
        except Exception as e:
            _LOGGER.error(f"Failed to get snapshot: {e}")
            
        return None


class LicensePlateDetector:
    """License plate detection using OpenCV and pattern matching."""
    
    def __init__(self, min_confidence: float = 0.8):
        self.min_confidence = min_confidence
        self.plate_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        )
        
    def detect_plates(self, image_data: bytes, detection_area: Optional[Dict] = None) -> List[str]:
        """Detect license plates in the image."""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return []
            
            # Apply detection area if specified
            if detection_area:
                x = int(detection_area.get('x', 0) * img.shape[1])
                y = int(detection_area.get('y', 0) * img.shape[0])
                w = int(detection_area.get('width', 1) * img.shape[1])
                h = int(detection_area.get('height', 1) * img.shape[0])
                img = img[y:y+h, x:x+w]
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect potential plate regions
            plates = self.plate_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 30)
            )
            
            detected_plates = []
            
            for (x, y, w, h) in plates:
                # Extract plate region
                plate_img = gray[y:y+h, x:x+w]
                
                # Enhance the image
                plate_img = cv2.bilateralFilter(plate_img, 11, 17, 17)
                
                # Apply threshold
                _, plate_img = cv2.threshold(
                    plate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                
                # Use simple OCR approach (in production, use pytesseract)
                plate_text = self._extract_text_simple(plate_img)
                
                if plate_text and self._is_valid_plate(plate_text):
                    detected_plates.append(plate_text)
            
            return detected_plates
            
        except Exception as e:
            _LOGGER.error(f"Plate detection failed: {e}")
            return []
    
    def _extract_text_simple(self, plate_img: np.ndarray) -> Optional[str]:
        """Simple text extraction - in production use pytesseract."""
        # This is a simplified approach
        # In a real implementation, you would use:
        # import pytesseract
        # return pytesseract.image_to_string(plate_img, config='--psm 8').strip()
        
        # For demo purposes, return a mock plate if image has sufficient contrast
        if plate_img.std() > 50:  # Basic contrast check
            return "ABC123"  # Mock detected plate
        return None
    
    def _is_valid_plate(self, plate_text: str) -> bool:
        """Validate if the detected text looks like a license plate."""
        if not plate_text or len(plate_text) < 4:
            return False
            
        # Basic patterns for common license plate formats
        patterns = [
            r'^[A-Z]{2,3}[0-9]{2,4}$',  # AB123, ABC1234
            r'^[0-9]{2,3}[A-Z]{2,3}$',  # 123AB, 12ABC
            r'^[A-Z0-9]{5,8}$',         # Mixed alphanumeric
        ]
        
        plate_clean = plate_text.upper().replace(' ', '').replace('-', '')
        
        for pattern in patterns:
            if re.match(pattern, plate_clean):
                return True
                
        return False


class ReolinkLicensePlateCoordinator:
    """Coordinator for managing the license plate detection system."""
    
    def __init__(self, hass: HomeAssistant, config: Dict[str, Any]):
        self.hass = hass
        self.config = config
        self.cameras = {}
        self.detector = LicensePlateDetector(
            min_confidence=config.get(CONF_MIN_CONFIDENCE, DEFAULT_CONFIDENCE)
        )
        self.last_detections = {}
        self.allowed_plates = set(config.get(CONF_ALLOWED_PLATES, []))
        self.cooldown_seconds = config.get(CONF_COOLDOWN_SECONDS, DEFAULT_COOLDOWN)
        
    async def async_setup(self) -> bool:
        """Set up the coordinator."""
        session = async_get_clientsession(self.hass)
        
        # Initialize cameras
        for camera_config in self.config.get(CONF_CAMERAS, []):
            camera_name = camera_config[CONF_CAMERA_NAME]
            
            api = ReolinkLicensePlateAPI(
                camera_config[CONF_HOST],
                camera_config.get(CONF_PORT, DEFAULT_PORT),
                camera_config[CONF_USERNAME],
                camera_config[CONF_PASSWORD]
            )
            
            if await api.authenticate(session):
                self.cameras[camera_name] = {
                    'api': api,
                    'config': camera_config
                }
                _LOGGER.info(f"Successfully connected to camera: {camera_name}")
            else:
                _LOGGER.error(f"Failed to connect to camera: {camera_name}")
                return False
        
        # Set up periodic detection
        async_track_time_interval(
            self.hass, self._periodic_detection, timedelta(seconds=5)
        )
        
        return True
    
    async def _periodic_detection(self, now: datetime) -> None:
        """Perform periodic license plate detection."""
        for camera_name, camera_data in self.cameras.items():
            try:
                await self._detect_plates_for_camera(camera_name)
            except Exception as e:
                _LOGGER.error(f"Detection failed for camera {camera_name}: {e}")
    
    async def _detect_plates_for_camera(self, camera_name: str) -> None:
        """Detect license plates for a specific camera."""
        camera_data = self.cameras.get(camera_name)
        if not camera_data:
            return
        
        api = camera_data['api']
        config = camera_data['config']
        
        # Get snapshot
        image_data = await api.get_snapshot()
        if not image_data:
            return
        
        # Detect plates
        detection_area = config.get(CONF_DETECTION_AREA)
        detected_plates = self.detector.detect_plates(image_data, detection_area)
        
        # Process detections
        for plate in detected_plates:
            await self._process_plate_detection(camera_name, plate, config)
    
    async def _process_plate_detection(
        self, camera_name: str, plate: str, config: Dict[str, Any]
    ) -> None:
        """Process a detected license plate."""
        now = dt_util.utcnow()
        
        # Check cooldown
        last_detection_key = f"{camera_name}_{plate}"
        last_detection = self.last_detections.get(last_detection_key)
        
        if (last_detection and 
            (now - last_detection).total_seconds() < self.cooldown_seconds):
            return
        
        self.last_detections[last_detection_key] = now
        
        # Log detection
        if config.get(CONF_LOG_ALL_DETECTIONS, False):
            _LOGGER.info(f"Detected plate {plate} on camera {camera_name}")
        
        # Check if plate is allowed
        is_allowed = plate.upper() in {p.upper() for p in self.allowed_plates}
        
        # Send notification
        if config.get(CONF_ENABLE_NOTIFICATIONS, True):
            await self._send_notification(camera_name, plate, is_allowed)
        
        # Trigger garage door if allowed and enabled
        if (is_allowed and 
            config.get(CONF_ENABLE_AUTO_GARAGE, False) and 
            config.get(CONF_GARAGE_DOOR_ENTITY)):
            
            await self._trigger_garage_door(config[CONF_GARAGE_DOOR_ENTITY])
        
        # Fire event
        self.hass.bus.async_fire(
            f"{DOMAIN}_plate_detected",
            {
                "camera": camera_name,
                "plate": plate,
                "allowed": is_allowed,
                "timestamp": now.isoformat()
            }
        )
    
    async def _send_notification(
        self, camera_name: str, plate: str, is_allowed: bool
    ) -> None:
        """Send a notification about the detection."""
        status = "Allowed" if is_allowed else "Unknown"
        message = f"License plate {plate} detected on {camera_name} ({status})"
        
        persistent_notification.async_create(
            self.hass,
            message,
            f"License Plate Detection - {camera_name}",
            f"{DOMAIN}_{camera_name}_{plate}"
        )
    
    async def _trigger_garage_door(self, garage_door_entity: str) -> None:
        """Trigger the garage door to open."""
        try:
            await self.hass.services.async_call(
                "cover", "open_cover", {"entity_id": garage_door_entity}
            )
            _LOGGER.info(f"Triggered garage door: {garage_door_entity}")
        except Exception as e:
            _LOGGER.error(f"Failed to trigger garage door: {e}")
    
    async def add_allowed_plate(self, plate: str, description: str = "") -> None:
        """Add a plate to the allowed list."""
        self.allowed_plates.add(plate.upper())
        _LOGGER.info(f"Added allowed plate: {plate}")
        
        # Update config entry
        # In a real implementation, you'd update the config entry data
        
    async def remove_allowed_plate(self, plate: str) -> None:
        """Remove a plate from the allowed list."""
        self.allowed_plates.discard(plate.upper())
        _LOGGER.info(f"Removed allowed plate: {plate}")


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the Reolink License Plate integration."""
    # This would typically be handled by config_entries
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Reolink License Plate from a config entry."""
    coordinator = ReolinkLicensePlateCoordinator(hass, entry.data)
    
    if not await coordinator.async_setup():
        raise ConfigEntryNotReady
    
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator
    
    # Set up platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    # Register services
    async def handle_add_plate(call: ServiceCall) -> None:
        plate = call.data["plate_number"]
        description = call.data.get("description", "")
        await coordinator.add_allowed_plate(plate, description)
    
    async def handle_remove_plate(call: ServiceCall) -> None:
        plate = call.data["plate_number"]
        await coordinator.remove_allowed_plate(plate)
    
    async def handle_trigger_detection(call: ServiceCall) -> None:
        camera_name = call.data["camera_name"]
        await coordinator._detect_plates_for_camera(camera_name)
    
    hass.services.async_register(
        DOMAIN, SERVICE_ADD_PLATE, handle_add_plate, schema=ADD_PLATE_SCHEMA
    )
    
    hass.services.async_register(
        DOMAIN, SERVICE_REMOVE_PLATE, handle_remove_plate, schema=REMOVE_PLATE_SCHEMA
    )
    
    hass.services.async_register(
        DOMAIN, SERVICE_TRIGGER_DETECTION, handle_trigger_detection,
        schema=TRIGGER_DETECTION_SCHEMA
    )
    
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
    
    return unload_ok
