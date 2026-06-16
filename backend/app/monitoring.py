import logging
from prometheus_client import Counter, Gauge

# Configure base logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("smartcampus_backend")

# Prometheus Custom Metrics
BADGE_SCANS = Counter(
    "smartcampus_badge_scans_total",
    "Total number of RFID badge scans",
    ["status", "room_id", "esp_ip"]
)

IOT_MESSAGES = Counter(
    "smartcampus_iot_messages_total",
    "Total number of IoT messages received from ESP32",
    ["ip_address", "status"]
)

SYSTEM_ERRORS = Counter(
    "smartcampus_system_errors_total",
    "Total system errors or exceptions",
    ["service", "error_type"]
)

DEVICE_STATUS = Gauge(
    "smartcampus_device_status",
    "Current status of IoT devices (1=online, 0=offline)",
    ["ip_address", "mac_address"]
)
