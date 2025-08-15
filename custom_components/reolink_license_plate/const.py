"""Constants for the Reolink License Plate Detection integration."""

DOMAIN = "reolink_license_plate"

# Configuration keys
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

# Default values
DEFAULT_PORT = 80
DEFAULT_SENSITIVITY = 0.7
DEFAULT_CONFIDENCE = 0.8
DEFAULT_COOLDOWN = 30

# Services
SERVICE_ADD_PLATE = "add_allowed_plate"
SERVICE_REMOVE_PLATE = "remove_allowed_plate"
SERVICE_TRIGGER_DETECTION = "trigger_detection"
SERVICE_OPEN_GARAGE = "open_garage_door"
SERVICE_SET_DETECTION_AREA = "set_detection_area"

# Events
EVENT_PLATE_DETECTED = f"{DOMAIN}_plate_detected"
EVENT_GARAGE_TRIGGERED = f"{DOMAIN}_garage_triggered"

# Platforms
PLATFORMS = ["sensor", "switch", "camera", "binary_sensor"]

# API endpoints
REOLINK_LOGIN_ENDPOINT = "/api.cgi?cmd=Login"
REOLINK_SNAPSHOT_ENDPOINT = "/cgi-bin/api.cgi"
REOLINK_LOGOUT_ENDPOINT = "/api.cgi?cmd=Logout"

# Detection settings
MIN_PLATE_WIDTH = 100
MIN_PLATE_HEIGHT = 30
MAX_DETECTIONS_PER_FRAME = 10

# Notification settings
NOTIFICATION_TIMEOUT = 300  # 5 minutes
NOTIFICATION_ID_PREFIX = f"{DOMAIN}_notification"
