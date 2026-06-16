#include <WiFi.h>
#include <WiFiManager.h>
#include <WebSocketsClient.h>
#include <MFRC522v2.h>
#include <MFRC522DriverSPI.h>
#include <MFRC522DriverPinSimple.h>
#include <MFRC522Debug.h>
#include <Preferences.h> // Ajout de la bibliothèque pour la mémoire Flash

// ==========================================
// CONFIGURATION DU SERVEUR MAÎTRE
// ==========================================
char server_ip[40] = "10.199.91.247";                       // Variable dynamique pour l'IP du serveur FastAPI
const uint16_t websocket_port = 8000;          // Port par défaut de Uvicorn/FastAPI
const char* websocket_path = "/ws/esp";        // Chemin défini dans ton APIRouter FastAPI

// Configuration RFID
MFRC522DriverPinSimple ss_pin(5);
MFRC522DriverSPI driver{ss_pin}; 
MFRC522 mfrc522{driver};         

// Configuration WebSocket et Mémoire
WebSocketsClient webSocket;
Preferences preferences; // Instance pour lire/écrire en mémoire flash
String macAddress = ""; // Stockera l'adresse MAC de l'ESP32

// Fonction de gestion des événements du Client WebSocket
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println(F("[WebSocket] Déconnecté du serveur maître !"));
      break;
    case WStype_CONNECTED: {
      Serial.printf("[WebSocket] Connecté au serveur maître à l'URL : %s\n", payload);
      
      // Envoi d'un premier message JSON pour s'enregistrer auprès du Master
      String initJson = "{\"mac_address\":\"" + macAddress + "\", \"status\":\"connected\"}";
      webSocket.sendTXT(initJson);
      break;
    }
    case WStype_TEXT:
      Serial.printf("[WebSocket] Message du maître (Ack) : %s\n", payload);
      break;
    case WStype_BIN:
    case WStype_ERROR:      
    case WStype_FRAGMENT_TEXT_START:
    case WStype_FRAGMENT_BIN_START:
    case WStype_FRAGMENT:
    case WStype_FRAGMENT_FIN:
      break;
  }
}

void setup() {
  Serial.begin(115200);  
  while (!Serial);       

  Serial.println(F("Démarrage de l'ESP32 en mode SLAVE..."));

  // --- GESTION DE LA MÉMOIRE ET DU PORTAIL WIFI ---
  
  // Ouvre l'espace "config" en mode lecture/écriture
  preferences.begin("config", false);
  
  // Récupère l'IP sauvegardée (si elle existe, sinon utilise la valeur de server_ip par défaut)
  String saved_ip = preferences.getString("server_ip", server_ip);
  strcpy(server_ip, saved_ip.c_str());
  Serial.print(F("IP du Master en mémoire : "));
  Serial.println(server_ip);

  // Initialisation de WiFiManager
  WiFiManager wm;
  
  // Ajout du champ personnalisé pour l'IP du serveur dans le portail captif
  WiFiManagerParameter custom_server_ip("server", "IP du Serveur Master", server_ip, 40);
  wm.addParameter(&custom_server_ip);
  
  Serial.println(F("Connexion au Wi-Fi via WiFiManager..."));
  bool res = wm.autoConnect("ESP32_RFID_AP"); 

  if(!res) {
    Serial.println(F("Échec de la connexion Wi-Fi. Redémarrage..."));
    ESP.restart();
  } 

  // Vérifie si l'utilisateur a modifié l'IP dans le portail captif
  if (String(server_ip) != String(custom_server_ip.getValue())) {
    Serial.println(F("Nouvelle IP détectée dans le portail. Sauvegarde en mémoire..."));
    strcpy(server_ip, custom_server_ip.getValue());
    preferences.putString("server_ip", server_ip);
  }
  
  // ------------------------------------------------

  Serial.println(F("Connecté au réseau Wi-Fi !"));
  Serial.print(F("Adresse IP locale : "));
  Serial.println(WiFi.localIP());

  // Récupération de l'adresse MAC
  macAddress = WiFi.macAddress();
  Serial.print(F("Adresse MAC : "));
  Serial.println(macAddress);

  // Démarrage du Client WebSocket avec l'IP dynamique
  webSocket.begin(server_ip, websocket_port, websocket_path);
  webSocket.setReconnectInterval(5000); // Reconnexion toutes les 5s en cas de perte
  webSocket.onEvent(webSocketEvent);
  
  // Initialisation du lecteur RFID
  mfrc522.PCD_Init();    
  MFRC522Debug::PCD_DumpVersionToSerial(mfrc522, Serial);
  Serial.println(F("En attente d'un badge RFID..."));
}

void loop() {
  // Maintient la connexion avec le serveur maître
  webSocket.loop();

  if (!mfrc522.PICC_IsNewCardPresent() || !mfrc522.PICC_ReadCardSerial()) {
    return;
  }

  // Extraction de l'UID en format hexadécimal pur (ex: "A1B2C3D4")
  String uidHex = "";
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    if(mfrc522.uid.uidByte[i] < 0x10) {
      uidHex += "0";
    }
    uidHex += String(mfrc522.uid.uidByte[i], HEX);
  }
  uidHex.toUpperCase(); 
  
  // Création du payload JSON avec l'adresse MAC et l'UID du badge
  String jsonPayload = "{\"mac_address\":\"" + macAddress + "\", \"rfid_uid\":\"" + uidHex + "\"}";
  
  Serial.println("Envoi au Master : " + jsonPayload);
  
  // Envoi du JSON au serveur Master
  webSocket.sendTXT(jsonPayload);

  // Délai pour éviter le multi-scan accidentel
  delay(2000);
}