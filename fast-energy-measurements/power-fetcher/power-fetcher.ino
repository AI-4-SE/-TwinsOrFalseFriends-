/**
   Author: ...
   Email: ...
*/
#include <Wire.h>
#include "bus-configuration.h"

#define HOST_FEEDBACK_RATE_SECONDS        1
elapsedMillis sinceHostListen;
int led = 13;

typedef struct measurement {
  char id; // 1 byte
  byte power [2]; // 2 bytes
} measurement;


typedef struct measurementWithTime {
  char id; // 1 byte
  byte power [2]; // 2 bytes
  unsigned int sampleTime; // 4 bytes, according to https://forum.pjrc.com/threads/36658-datatypes-bit-depth?p=114137&viewfull=1
} measurementWithTime;

template <typename T>
unsigned int USB_writeBytes(const T& value) {
  const byte * p = (const byte*) &value;
  unsigned int i;
  byte newlineMask = 0x00;
  for (i = 0; i < sizeof(value); i++) {
    if (*p == 0x0A) {
      newlineMask |= 1 << i;
      Serial.write(0x00);
      p++;
    } else {
      Serial.write(*p++);
    }
  }
  Serial.write(newlineMask);
  Serial.println();
  return i;
}


unsigned int USB_write_struct(const measurementWithTime& value) {
  //const byte * p = (const byte*) &value;
  unsigned int i = 0;

  byte id = *((unsigned char *)&value.id );
  byte msb = *((unsigned char *)&value.power + 0 );
  byte lsb = *((unsigned char *)&value.power + 1 );

  byte t1 = *((unsigned char *)&value.sampleTime + 0 );
  byte t2 = *((unsigned char *)&value.sampleTime + 1 );
  byte t3 = *((unsigned char *)&value.sampleTime + 2 );
  byte t4 = *((unsigned char *)&value.sampleTime + 3 );

  byte payload [7]= {id, msb, lsb, t1, t2, t3, t4};
  //byte payload [7]= {id, '\n', '\n', t1, t2, t3, t4};
  byte newlineMask = 0x80; // only set first bit to avoid \n if second-to-last package is \n

  for (i = 0; i < sizeof(payload); i++) {
    //if (payload[i] == 0x0A) {
    if (payload[i] == '\n') {
      newlineMask |= 1 << i;
      Serial.write(0x00);
    } else {
      newlineMask |= 0 << i;
      Serial.write(payload[i]);
    }
  }

  //unsigned char byte id = *((unsigned char *)&x + i);
  Serial.write(newlineMask);
  Serial.write(0x0D); // \r
  Serial.write(0x0A); // \n
  //TODO
  // println weg
  // -------Serial.send_now()
  // byte als array senden


  return i;
}



void writeCalibrationToBus(TwoWire wire, int INAs [16], int confs[16], int cals [16], bool online [16]) {
    for (int i = 0; i < 16; ++i) {
      if (INAs[i] != NO_DEVICE) {
           //Write Configuration to INA226
          wire.beginTransmission(INAs[i]);
          wire.write(CONFIGURATION_REGISTER);
          wire.write((confs[i] >> 8) & 0xFF);
          wire.write(confs[i] & 0xFF);
          wire.endTransmission();

          //Write Calibration to INA226
          wire.beginTransmission(INAs[i]);
          wire.write(CALIBRATION_REGISTER);
          wire.write((cals[i] >> 8) & 0xFF);
          wire.write(cals[i] & 0xFF);
          wire.endTransmission();

          //Set INA226 Register-Pointer to the Power-Register
          wire.beginTransmission(INAs[i]);
          wire.write(POWER_REGISTER);
          wire.endTransmission();

          // check if INA is online
          bool readOnline = isOnline(wire, INAs[i]);
          online[i] = readOnline;
      }
  }
}

void writeCalibration(){
  writeCalibrationToBus(Wire, INAs0, CONF0, CALs0, ONLINE0);
  writeCalibrationToBus(Wire1, INAs1, CONF1, CALs1, ONLINE1);
  writeCalibrationToBus(Wire2, INAs2, CONF2, CALs2, ONLINE2);
}

void setup() {
  pinMode(led, OUTPUT);

  blink();


  Serial.begin(1); // always 12 mbit/s for USB serial
  while (!Serial);
  Wire.begin();
  Wire.setClock(BUS_CLOCK_FREQUENCY);

  Wire1.begin();
  Wire1.setClock(BUS_CLOCK_FREQUENCY);

  Wire2.begin();
  Wire2.setClock(BUS_CLOCK_FREQUENCY);
  //Serial.println("Running calibration for INAs.");


  writeCalibration();
}


bool isOnline(TwoWire wire, int address) {
  wire.requestFrom(address, 2);
  if (wire.available() == 2) {
    int raw_data;
    raw_data = wire.read();
    raw_data = raw_data << 8;
    raw_data = raw_data + wire.read();
    //Serial.print(raw_data);
    return true;
  } else {
    //Something went wrong during the I2C-Communication
    return false;
  }
}

int isAvailable(int address) {

  Wire.beginTransmission(address);
  Wire.write(MASK_ENABLE_REGISTER);
  Wire.endTransmission();

  Wire.requestFrom(address, 2);
  int result = -1;
  if (Wire.available() == 2) {
    //int raw_data;
    //raw_data = Wire.read();
    //raw_data = raw_data << 8;
    //raw_data = raw_data + Wire.read();
    //Serial.print(raw_data);

    // MISSING : read ready flag and return true if it is set.

    result = 1;
  } else {
    //Something went wrong during the I2C-Communication
    result = 0;
  }
    Wire.beginTransmission(address);
    Wire.write(POWER_REGISTER);
    Wire.endTransmission();
    return result;
}



bool should_reconfigure() {
  bool do_reconfigure;
  int avail_bytes = Serial.available();
  if(avail_bytes > 0) {
    do_reconfigure = true;
    for (int n = 0; n < avail_bytes; n++) {
      Serial.read();
    }
    for (int i = 0; i < 20; i ++) {
      Serial.write(0xFF);
    }
  } else {
  do_reconfigure = false;
  }
  return do_reconfigure;
}

void clearConfCalib() {
    for (int i = 0; i < 16; i++) {
      ONLINE0[i] = 0;
      ONLINE1[i] = 0;
      ONLINE2[i] = 0;
      COMPONENTS0[i] = NO_DEVICE;
      COMPONENTS1[i] = NO_DEVICE;
      COMPONENTS2[i] = NO_DEVICE;
      CALs0[i] = 0x0000;
      CALs1[i] = 0x0000;
      CALs2[i] = 0x0000;
      CONF0[i] = 0x0000;
      CONF1[i] = 0x0000;
      CONF2[i] = 0x0000;
      INAs0[i] = 0x00;
      INAs1[i] = 0x00;
      INAs2[i] = 0x00;
  }
}

void reconfigure() {
  digitalWrite(led, LOW);
  delay(50);
  //Serial.println("Commencing Reconfiguration. Waiting for Config.");
  //const byte idByte = 0x40;
  digitalWrite(led, HIGH);   // turn the LED on (HIGH is the voltage level)
  clearConfCalib();

  byte idByte = Serial.read();
  while (idByte != 0x00) {
    const char deviceId = (char) idByte;
    const byte busByte = Serial.read();
    const short busNo = (short) busByte;
    const int busAddress =  (int) Serial.read();
    ///const byte calibOrAveragingByte = Serial.read();
    const byte msbCalib = Serial.read();
    const byte lsbCalib = Serial.read();
    const byte msbConf = Serial.read();
    const byte lsbConf = Serial.read();
    int calib = lsbCalib | msbCalib << 8;
    int conf = lsbConf | msbConf << 8;

    if (busNo == 0) {
      addDevice(COMPONENTS0, INAs0, CALs0, CONF0, deviceId, busAddress, calib, conf);
    } else if (busNo == 1) {
      addDevice(COMPONENTS1, INAs1, CALs1, CONF1, deviceId, busAddress, calib, conf);
    } else if (busNo == 2) {
      addDevice(COMPONENTS2, INAs2, CALs2, CONF2, deviceId, busAddress, calib, conf);
    }
    /*Serial.print("deviceId <");
    Serial.print(deviceId);
    Serial.print("> & busAddress <");
    Serial.print(busAddress);
    Serial.print("> & busNo <");
    Serial.print(busNo);
    Serial.print("> & calib <");
    Serial.print(calib);
    Serial.print("> & conf <");
    Serial.print(conf);
    Serial.print(">");

    Serial.println();*/
    delay(100);
    idByte = Serial.read();
  }
  delay(50);
  //blink_short();
  writeCalibration();

  int onlineDevices = getNumberOnlineDevices();
  //Serial.println(String(registeredDevices));

  blink_short();
  Serial.write(onlineDevices);
  write_found_IDs();
  delay(200);
  digitalWrite(led, LOW);    // turn the LED off by making the voltage LOW
}

void write_found_IDs(){
  for (int i = 0; i < 16; i++) {
    if (ONLINE0[i] > 0 ) {
      Serial.write((byte) COMPONENTS0[i]);
    }
    if (ONLINE1[i] > 0 ) {
      Serial.write((byte) COMPONENTS1[i]);
    }
    if (ONLINE2[i] > 0 ) {
      Serial.write((byte) COMPONENTS2[i]);
    }
  }
  Serial.write((byte) 0);
}


void blink() {
  for (int i = 0; i < 3; i ++) {
    digitalWrite(led, LOW);
    delay(300);
    digitalWrite(led, HIGH);
    delay(300);
  }
}

void blink_short() {
  for (int i = 0; i < 3; i ++) {
    digitalWrite(led, LOW);
    delay(75);
    digitalWrite(led, HIGH);
    delay(75);
  }
}

int getNumberOnlineDevices() {
  int foundDevices = 0;
  for (int i = 0; i < 16; i++) {
    if (ONLINE0[i] > 0 ) {
      foundDevices ++;
    }
    if (ONLINE1[i] > 0 ) {
      foundDevices ++;
    }
    if (ONLINE2[i] > 0 ) {
      foundDevices ++;
    }
  }
  return foundDevices;
}

void addDevice(char componentIDs[16], int componentAddresses [16], int cals [16], int confs [16], const char deviceId, const int busAddress, const int cal, const int conf) {
  for (int i = 0; i < 16; i++) {
    if (componentAddresses[i] == NO_DEVICE) {
      // found position to insert new device
      componentAddresses[i] = busAddress;
      componentIDs[i] = deviceId;
      cals[i] = cal;
      confs[i] = conf;
      break;
    }
  }
}


void loop() {
  //sendAllVals();
  process_bus(Wire, COMPONENTS0, INAs0, CONF0, CALs0, ONLINE0);
  process_bus(Wire1, COMPONENTS1, INAs1, CONF1, CALs1, ONLINE1);
  process_bus(Wire2, COMPONENTS2, INAs2, CONF2, CALs2, ONLINE2);

  if (sinceHostListen > HOST_FEEDBACK_RATE_SECONDS * 1000) {      // "sinceHostListen" auto-increases
    sinceHostListen = 0;

    digitalWrite(led, HIGH);
    if (should_reconfigure()) {
      // HOST WANTS TO RECONFIGURE
      reconfigure();
    }
  }

  //delay(100);
}

void process_bus(TwoWire wire, char componentIDs[16], int INAs [16], int confs[16], int cals [16], bool online [16]) {
    byte raw_data1;
  byte raw_data2;
  String message;

  for (int i = 0; i < 16; ++i) {
    if (online[i] == 1) {
      const char id = componentIDs[i];
      wire.requestFrom(INAs[i], 2);
      if (wire.available() == 2) {
        unsigned int sampleTime = micros();
        raw_data1 = wire.read();
        raw_data2 = wire.read();
        const measurementWithTime m = {id, {raw_data1, raw_data2}, sampleTime};
        USB_write_struct(m);
      } else {
        //Something went wrong during the I2C-Communication
        unsigned int sampleTime = micros();
        const measurementWithTime m = {id, {0, 0}, sampleTime};
        USB_write_struct(m);
      }
    }
  }
}

void sendAllVals() {
  byte raw_data1;
  byte raw_data2;
  String message;

  for (int i = 0; i < 16; ++i) {
    if (ONLINE0[i] == 1) {

      //isAvailable(INAs0[0]);

      const char id = COMPONENTS0[i];
      Wire.requestFrom(INAs0[i], 2);
      if (Wire.available() == 2) {
        unsigned int sampleTime = micros();
        raw_data1 = Wire.read();
        raw_data2 = Wire.read();
        const measurementWithTime m = {id, {raw_data1, raw_data2}, sampleTime};
        //for (int n = 0; n < 10; n++) {
          USB_write_struct(m);
        //}

        //const measurement m = {id, {raw_data1, raw_data2}};
        //USB_write_struct_no_time(m);
      } else {
        //Something went wrong during the I2C-Communication
        unsigned int sampleTime = micros();
        const measurementWithTime m = {id, {0, 0}, sampleTime};
        //for (int n = 0; n < 10; n++) {
          USB_write_struct(m);
        //}
        //const measurement m = {id, {0, 0}};
        //USB_write_struct_no_time(m);

      }
    }
  }
}
