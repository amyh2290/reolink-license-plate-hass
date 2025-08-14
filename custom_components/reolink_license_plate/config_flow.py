"""
Config flow for Reolink License Plate Detection integration.
"""
import logging
from typing import Any, Dict, Optional

import aiohttp
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_HOST, CONF_PASSWORD, CONF_PORT, CONF_USERNAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers import config_validation as cv

from .const import (
    DOMAIN,
    CONF_CAMERAS,
    CONF_CAMERA_NAME,
    CONF_DETECTION_AREA,
    CONF_ALLOWED_PLATES,
    CONF_GARAGE_DOOR_ENTITY,
    CONF_DETECTION_SENSITIVITY,
    CONF_MIN_CONFIDENCE,
    CONF_COOLDOWN_SECONDS,
    CONF_ENABLE_NOTIFICATIONS,
    CONF_ENABLE_AUTO_GARAGE,
    CONF_LOG_ALL_DETECTIONS,
    DEFAULT_PORT,
    DEFAULT_SENSITIVITY,
    DEFAULT_CONFIDENCE,
    DEFAULT_COOLDOWN,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema({
    vol.Required("integration_name", default="Reolink License Plate"): cv.string,
})

STEP_CAMERA_DATA_SCHEMA = vol.Schema({
    vol.Required(CONF_CAMERA_NAME): cv.string,
    vol.Required(CONF_HOST): cv.string,
    vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
    vol.Required(CONF_USERNAME): cv.string,
    vol.Required(CONF_PASSWORD): cv.string,
})

STEP_DETECTION_DATA_SCHEMA = vol.Schema({
    vol.Optional(CONF_DETECTION_SENSITIVITY, default=DEFAULT_SENSITIVITY): vol.All(
        vol.Coerce(float), vol.Range(min=0.1, max=1.0)
    ),
    vol.Optional(CONF_MIN_CONFIDENCE, default=DEFAULT_CONFIDENCE): vol.All(
        vol.Coerce(float), vol.Range(min=0.1, max=1.0)
    ),
    vol.Optional(CONF_COOLDOWN_SECONDS, default=DEFAULT_COOLDOWN): vol.All(
        vol.Coerce(int), vol.Range(min=5, max=300)
    ),
})

STEP_PLATES_DATA_SCHEMA = vol.Schema({
    vol.Optional(CONF_ALLOWED_PLATES, default=""): cv.string,
})

STEP_GARAGE_DATA_SCHEMA = vol.Schema({
    vol.Optional(CONF_GARAGE_DOOR_ENTITY): cv.string,
    vol.Optional(CONF_ENABLE_AUTO_GARAGE, default=False): cv.boolean,
    vol.Optional(CONF_ENABLE_NOTIFICATIONS, default=True): cv.boolean,
    vol.Optional(CONF_LOG_ALL_DETECTIONS, default=False): cv.boolean,
})

STEP_AREA_DATA_SCHEMA = vol.Schema({
    vol.Optional("enable_detection_area", default=False): cv.boolean,
    vol.Optional("area_x", default=0.0): vol.All(
        vol.Coerce(float), vol.Range(min=0.0, max=1.0)
    ),
    vol.Optional("area_y", default=0.0): vol.All(
        vol.Coerce(float), vol.Range(min=0.0, max=1.0)
    ),
    vol.Optional("area_width", default=1.0): vol.All(
        vol.Coerce(float), vol.Range(min=0.1, max=1.0)
    ),
    vol.Optional("area_height", default=1.0): vol.All(
        vol.Coerce(float), vol.Range(min=0.1, max=1.0)
    ),
})


async def validate_camera_connection(
    hass: HomeAssistant, data: Dict[str, Any]
) -> Dict[str, str]:
    """Validate camera connection."""
    session = async_get_clientsession(hass)
    
    auth_url = f"http://{data[CONF_HOST]}:{data[CONF_PORT]}/api.cgi?cmd=Login"
    auth_data = {
        "cmd": "Login",
        "param": {
            "User": {
                "userName": data[CONF_USERNAME],
                "password": data[CONF_PASSWORD]
            }
        }
    }
    
    try:
        async with session.post(auth_url, json=[auth_data], timeout=10) as resp:
            if resp.status == 200:
                result = await resp.json()
                if result[0]["code"] == 0:
                    return {"title": f"Camera {data[CONF_CAMERA_NAME]}"}
    except aiohttp.ClientError:
        pass
    except Exception as e:
        _LOGGER.error(f"Unexpected error: {e}")
    
    raise ValueError("Cannot connect to camera")


class ReolinkLicensePlateConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Reolink License Plate Detection."""

    VERSION = 1

    def __init__(self):
        """Initialize the config flow."""
        self.cameras = []
        self.config_data = {}
        self.current_camera = {}

    async def async_step_user(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle the initial step."""
        if user_input is not None:
            self.config_data["integration_name"] = user_input["integration_name"]
            return await self.async_step_camera()

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            description_placeholders={
                "integration_docs": "https://github.com/example/reolink-license-plate"
            },
        )

    async def async_step_camera(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle camera configuration step."""
        errors = {}

        if user_input is not None:
            try:
                info = await validate_camera_connection(self.hass, user_input)
                self.current_camera = user_input.copy()
                return await self.async_step_detection_area()
            except ValueError:
                errors["base"] = "cannot_connect"
            except Exception:
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="camera",
            data_schema=STEP_CAMERA_DATA_SCHEMA,
            errors=errors,
            description_placeholders={
                "camera_count": str(len(self.cameras) + 1)
            },
        )

    async def async_step_detection_area(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle detection area configuration."""
        if user_input is not None:
            if user_input["enable_detection_area"]:
                self.current_camera[CONF_DETECTION_AREA] = {
                    "x": user_input["area_x"],
                    "y": user_input["area_y"],
                    "width": user_input["area_width"],
                    "height": user_input["area_height"],
                }
            
            return await self.async_step_detection_settings()

        return self.async_show_form(
            step_id="detection_area",
            data_schema=STEP_AREA_DATA_SCHEMA,
            description_placeholders={
                "camera_name": self.current_camera[CONF_CAMERA_NAME]
            },
        )

    async def async_step_detection_settings(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle detection settings configuration."""
        if user_input is not None:
            self.current_camera.update(user_input)
            self.cameras.append(self.current_camera)
            self.current_camera = {}
            
            return await self.async_step_more_cameras()

        return self.async_show_form(
            step_id="detection_settings",
            data_schema=STEP_DETECTION_DATA_SCHEMA,
        )

    async def async_step_more_cameras(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Ask if user wants to add more cameras."""
        if user_input is not None:
            if user_input.get("add_another_camera"):
                return await self.async_step_camera()
            else:
                return await self.async_step_allowed_plates()

        return self.async_show_form(
            step_id="more_cameras",
            data_schema=vol.Schema({
                vol.Optional("add_another_camera", default=False): cv.boolean,
            }),
            description_placeholders={
                "camera_count": str(len(self.cameras))
            },
        )

    async def async_step_allowed_plates(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle allowed plates configuration."""
        if user_input is not None:
            plates_text = user_input.get(CONF_ALLOWED_PLATES, "")
            if plates_text:
                # Parse comma-separated or line-separated plates
                plates = [
                    plate.strip().upper() 
                    for plate in plates_text.replace(",", "\n").split("\n")
                    if plate.strip()
                ]
                self.config_data[CONF_ALLOWED_PLATES] = plates
            else:
                self.config_data[CONF_ALLOWED_PLATES] = []
            
            return await self.async_step_garage_settings()

        return self.async_show_form(
            step_id="allowed_plates",
            data_schema=STEP_PLATES_DATA_SCHEMA,
        )

    async def async_step_garage_settings(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle garage door settings."""
        errors = {}

        if user_input is not None:
            # Validate garage door entity if provided
            garage_entity = user_input.get(CONF_GARAGE_DOOR_ENTITY)
            if garage_entity:
                if not self.hass.states.get(garage_entity):
                    errors["base"] = "garage_entity_not_found"
                elif not garage_entity.startswith("cover."):
                    errors["base"] = "garage_entity_invalid"

            if not errors:
                self.config_data.update(user_input)
                return await self.async_step_final()

        # Get available cover entities for the dropdown
        cover_entities = [
            entity_id for entity_id in self.hass.states.async_entity_ids("cover")
            if "garage" in entity_id.lower() or "door" in entity_id.lower()
        ]

        garage_schema = STEP_GARAGE_DATA_SCHEMA
        if cover_entities:
            garage_schema = vol.Schema({
                vol.Optional(CONF_GARAGE_DOOR_ENTITY): vol.In(
                    [""] + cover_entities
                ),
                vol.Optional(CONF_ENABLE_AUTO_GARAGE, default=False): cv.boolean,
                vol.Optional(CONF_ENABLE_NOTIFICATIONS, default=True): cv.boolean,
                vol.Optional(CONF_LOG_ALL_DETECTIONS, default=False): cv.boolean,
            })

        return self.async_show_form(
            step_id="garage_settings",
            data_schema=garage_schema,
            errors=errors,
        )

    async def async_step_final(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Create the config entry."""
        # Combine all configuration
        final_config = {
            CONF_CAMERAS: self.cameras,
            **self.config_data
        }

        return self.async_create_entry(
            title=self.config_data.get("integration_name", "Reolink License Plate"),
            data=final_config,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return OptionsFlowHandler(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for the integration."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry
        self.options = dict(config_entry.options)

    async def async_step_init(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return await self.async_step_option_type()

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required("option_type"): vol.In([
                    "cameras",
                    "plates",
                    "detection",
                    "garage",
                    "notifications"
                ]),
            }),
        )

    async def async_step_option_type(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle option type selection."""
        option_type = user_input["option_type"]
        
        if option_type == "cameras":
            return await self.async_step_modify_cameras()
        elif option_type == "plates":
            return await self.async_step_modify_plates()
        elif option_type == "detection":
            return await self.async_step_modify_detection()
        elif option_type == "garage":
            return await self.async_step_modify_garage()
        elif option_type == "notifications":
            return await self.async_step_modify_notifications()

    async def async_step_modify_cameras(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Modify camera settings."""
        if user_input is not None:
            self.options.update(user_input)
            return self.async_create_entry(title="", data=self.options)

        current_cameras = self.config_entry.data.get(CONF_CAMERAS, [])
        cameras_text = "\n".join([
            f"{cam[CONF_CAMERA_NAME]} - {cam[CONF_HOST]}:{cam.get(CONF_PORT, DEFAULT_PORT)}"
            for cam in current_cameras
        ])

        return self.async_show_form(
            step_id="modify_cameras",
            data_schema=vol.Schema({
                vol.Optional("cameras_info", default=cameras_text): cv.string,
                vol.Optional("add_new_camera", default=False): cv.boolean,
                vol.Optional("remove_camera"): vol.In([
                    cam[CONF_CAMERA_NAME] for cam in current_cameras
                ] + [""]),
            }),
        )

    async def async_step_modify_plates(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Modify allowed plates."""
        if user_input is not None:
            plates_text = user_input.get("allowed_plates", "")
            if plates_text:
                plates = [
                    plate.strip().upper() 
                    for plate in plates_text.replace(",", "\n").split("\n")
                    if plate.strip()
                ]
                self.options[CONF_ALLOWED_PLATES] = plates
            else:
                self.options[CONF_ALLOWED_PLATES] = []
            
            return self.async_create_entry(title="", data=self.options)

        current_plates = self.config_entry.data.get(CONF_ALLOWED_PLATES, [])
        plates_text = "\n".join(current_plates)

        return self.async_show_form(
            step_id="modify_plates",
            data_schema=vol.Schema({
                vol.Optional("allowed_plates", default=plates_text): cv.string,
            }),
        )

    async def async_step_modify_detection(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Modify detection settings."""
        if user_input is not None:
            self.options.update(user_input)
            return self.async_create_entry(title="", data=self.options)

        current_data = self.config_entry.data
        
        return self.async_show_form(
            step_id="modify_detection",
            data_schema=vol.Schema({
                vol.Optional(
                    CONF_DETECTION_SENSITIVITY, 
                    default=current_data.get(CONF_DETECTION_SENSITIVITY, DEFAULT_SENSITIVITY)
                ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=1.0)),
                vol.Optional(
                    CONF_MIN_CONFIDENCE,
                    default=current_data.get(CONF_MIN_CONFIDENCE, DEFAULT_CONFIDENCE)
                ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=1.0)),
                vol.Optional(
                    CONF_COOLDOWN_SECONDS,
                    default=current_data.get(CONF_COOLDOWN_SECONDS, DEFAULT_COOLDOWN)
                ): vol.All(vol.Coerce(int), vol.Range(min=5, max=300)),
            }),
        )

    async def async_step_modify_garage(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Modify garage door settings."""
        if user_input is not None:
            self.options.update(user_input)
            return self.async_create_entry(title="", data=self.options)

        current_data = self.config_entry.data
        cover_entities = [
            entity_id for entity_id in self.hass.states.async_entity_ids("cover")
            if "garage" in entity_id.lower() or "door" in entity_id.lower()
        ]

        return self.async_show_form(
            step_id="modify_garage",
            data_schema=vol.Schema({
                vol.Optional(
                    CONF_GARAGE_DOOR_ENTITY,
                    default=current_data.get(CONF_GARAGE_DOOR_ENTITY, "")
                ): vol.In([""] + cover_entities),
                vol.Optional(
                    CONF_ENABLE_AUTO_GARAGE,
                    default=current_data.get(CONF_ENABLE_AUTO_GARAGE, False)
                ): cv.boolean,
            }),
        )

    async def async_step_modify_notifications(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Modify notification settings."""
        if user_input is not None:
            self.options.update(user_input)
            return self.async_create_entry(title="", data=self.options)

        current_data = self.config_entry.data

        return self.async_show_form(
            step_id="modify_notifications",
            data_schema=vol.Schema({
                vol.Optional(
                    CONF_ENABLE_NOTIFICATIONS,
                    default=current_data.get(CONF_ENABLE_NOTIFICATIONS, True)
                ): cv.boolean,
                vol.Optional(
                    CONF_LOG_ALL_DETECTIONS,
                    default=current_data.get(CONF_LOG_ALL_DETECTIONS, False)
                ): cv.boolean,
            }),
        )