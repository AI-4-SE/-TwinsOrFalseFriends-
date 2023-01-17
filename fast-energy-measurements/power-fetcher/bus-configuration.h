
//Define I2C-Bus Clock Frequency
//#define BUS_CLOCK_FREQUENCY         400000
//#define BUS_CLOCK_FREQUENCY         100000
//#define BUS_CLOCK_FREQUENCY            800000
#define BUS_CLOCK_FREQUENCY           1000000          
//#define BUS_CLOCK_FREQUENCY           5000000
//#define BUS_CLOCK_FREQUENCY         2940000

//Define INA226-Registers
#define CONFIGURATION_REGISTER      0x00
#define SHUNT_VOLTAGE_REGISTER      0x01
#define BUS_VOLTAGE_REGISTER        0x02
#define POWER_REGISTER              0x03
#define CURRENT_REGISTER            0x04
#define CALIBRATION_REGISTER        0x05
#define MASK_ENABLE_REGISTER        0x06
#define ALERT_LIMIT_REGISTER        0x07
#define MANUFACTURER_ID_REGISTER    0xFE
#define DIE_ID_REGISTER             0xFF

//Define Configuration
//0x4007 -> ConversionTime: 140 microseconds, Averaging: 1
//#define CONFIGURATION_MSB           0x40
//#define CONFIGURATION_LSB           0x07
//0x4007 -> ConversionTime: 140 microseconds, Averaging: 4
//#define CONFIGURATION_MSB           0x42
//#define CONFIGURATION_LSB           0x07

//0x0007 -> ConversionTime: 140 microseconds, Averaging: 1 -- checked the manual!!
#define CONFIGURATION_MSB           0x00
#define CONFIGURATION_LSB           0x07

//Define Addresses of measured components
#define CPU_12V_1                   0x40  //64
#define CPU_12V_2                   0x41  //65
#define SSD_5V                      0x42  //66
#define SSD_12V                     0x43  //67
#define MOBO_5V_1                   0x44  //68
#define MOBO_5V_2                   0x45  //69


#define NO_DEVICE 0x00

int INAs0[16] = { CPU_12V_1,
                 CPU_12V_2,
                 SSD_5V,
                 SSD_12V,
                 MOBO_5V_1,
                 MOBO_5V_2,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE
               };

// calibration speeed and averaging
int CONF0[16] = {
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007
  };
               

//Define Calibration for each INA226
int CALs0[16] = { 0x0D55,
                 0x0D55,
                 0x0A00,
                 0x0A00,
                 0x3348,
                 0x3348,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE
               };


//const char *COMPONENTNAMES0[6] = { "CPU_12V_1",
                                   //"CPU_12V_2",
                                   //"SSD_5V",
                                   //"SSD_12V",
                                   //"MOBO_5V_1",
                                   //"MOBO_5V_2",
                                 //};

char COMPONENTS0[16] = { 'H',
                               'I',
                               'J',
                               'K',
                               'L',
                               'M',     
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE
                             };

bool ONLINE0[16] = { 0,
                    0,
                    0,
                    0,
                    0,
                    0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0
                  };



//BUS 1 Adresses
#define MOBO_3_3V_1                 0x40  //64
#define MOBO_3_3V_2                 0x41  //65
#define MOBO_3_3V_3                 0x42  //66
#define MOBO_3_3V_4                 0x43  //67
#define MOBO_12V_1                  0x44  //68
#define MOBO_12V_2                  0x45  //69


int INAs1[16] = { MOBO_3_3V_1,
                 MOBO_3_3V_2,
                 MOBO_3_3V_3,
                 MOBO_3_3V_4,
                 MOBO_12V_1,
                 MOBO_12V_2,
                 
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE
               };

// calibration speeed and averaging
int CONF1[16] = {
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007
  };

//Define Calibration for each INA226
int CALs1[16] = { 0x0F62,
                 0x0F62,
                 0x0F62,
                 0x0F62,
                 0x3348,
                 0x3348,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE
               };


//const char *COMPONENTNAMES1[6] = { "MOBO_3_3V_1",
                               //"MOBO_3_3V_2",
                               //"MOBO_3_3V_3",
                               //"MOBO_3_3V_4",
                               //"MOBO_12V_1",
                               //"MOBO_12V_2"
                             //};

char COMPONENTS1[16] = { 'A',
                               'B',
                               'C',
                               'D',
                               'E',
                               'F',        
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE
                             };


bool ONLINE1[16] = { 0,
                    0,
                    0,
                    0,
                    0,
                    0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0
                  };






int INAs2[16] = { NO_DEVICE, NO_DEVICE, NO_DEVICE,
                 NO_DEVICE, NO_DEVICE, NO_DEVICE,
                 NO_DEVICE, NO_DEVICE, NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE
               };

// calibration speeed and averaging
int CONF2[16] = {
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007,
    0x0007
  };

//Define Calibration for each INA226
int CALs2[16] = { NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE,
                 NO_DEVICE
               };

char COMPONENTS2[16] = { NO_DEVICE, NO_DEVICE, NO_DEVICE, 
                        NO_DEVICE, NO_DEVICE, NO_DEVICE,        
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE,
                               NO_DEVICE
                             };


bool ONLINE2[16] = { 0,
                    0,
                    0,
                    0,
                    0,
                    0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0
                  };                  
