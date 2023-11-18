#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <linux/i2c-dev.h>


#define PULSE_PERIOD 500
#define CMD_PERIOD 4100

#define BACKLIGHT 8
#define DATA 1
static int iBackLight = BACKLIGHT;
static int file_i2c = -1;

static void WriteCommand(unsigned char ucCMD) {

	unsigned char uc; 
	uc = (ucCMD & 0xf0) | iBackLight; 
	write(file_i2c, &uc, 1);
	usleep(PULSE_PERIOD); 
	uc |= 4; 
	write(file_i2c, &uc, 1);
	usleep(PULSE_PERIOD);
	uc &= ~4; 
	write(file_i2c, &uc, 1);
	usleep(CMD_PERIOD);
	uc = iBackLight | (ucCMD << 4); 
	write(file_i2c, &uc, 1);
	usleep(PULSE_PERIOD);
	uc |= 4; // enable pulse
	write(file_i2c, &uc, 1);
	usleep(PULSE_PERIOD);
	uc &= ~4; // toggle pulse
	write(file_i2c, &uc, 1);
	usleep(CMD_PERIOD);

}

int lcd2004Control(int bBacklight, int bCursor, int bBlink) {
	unsigned char ucCMD = 0xc; // display control

	if (file_i2c < 0)
		return 1;
	iBackLight = (bBacklight) ? BACKLIGHT : 0;
	if (bCursor)
		ucCMD |= 2;
	if (bBlink)
		ucCMD |= 1;
	WriteCommand(ucCMD);
 	
	return 0;
}

int lcd2004WriteString(char *text) {
	unsigned char ucTemp[2];
	int i = 0;

	if (file_i2c < 0 || text == NULL)
		return 1;

	while (i<20 && *text) {
		ucTemp[0] = iBackLight | DATA | (*text & 0xf0);
		write(file_i2c, ucTemp, 1);
		usleep(PULSE_PERIOD);
		ucTemp[0] |= 4; // pulse E
		write(file_i2c, ucTemp, 1);
		usleep(PULSE_PERIOD);
		ucTemp[0] &= ~4;
		write(file_i2c, ucTemp, 1);
		usleep(PULSE_PERIOD);
		ucTemp[0] = iBackLight | DATA | (*text << 4);
		write(file_i2c, ucTemp, 1);
		ucTemp[0] |= 4; // pulse E

		write(file_i2c, ucTemp, 1);
		usleep(PULSE_PERIOD);
		ucTemp[0] &= ~4;
		write(file_i2c, ucTemp, 1);
		usleep(CMD_PERIOD);
		text++;
		i++;
	}
	return 0;
}

int lcd2004Clear(void) {
	if (file_i2c < 0)
		return 1;
	WriteCommand(0x01);
	return 0;
}

int lcd2004Init(int iChannel, int iAddr) {
	char szFile[80];
	int rc;

	sprintf(szFile, "/dev/i2c-%d", iChannel);
	file_i2c = open(szFile, O_RDWR);
	if (file_i2c < 0)
	{
		fprintf(stderr, "Error opening i2c device;\n");
		return 1;
	}
	rc = ioctl(file_i2c, I2C_SLAVE, iAddr);
	if (rc < 0)
	{
		close(file_i2c);
		fprintf(stderr, "Error setting I2C device address\n");
		return 1;
	}
	iBackLight = BACKLIGHT; // turn on backlight
	WriteCommand(0x02); // Set 4-bit mode of the LCD controller
	WriteCommand(0x28); // 2 lines, 5x8 dot matrix
	WriteCommand(0x0c); // display on, cursor off
	WriteCommand(0x06); // inc cursor to right when writing and don't scroll
	WriteCommand(0x80); // set cursor to row 1, column 1
	lcd2004Clear();	    // clear the memory

	return 0;
}

int lcd2004SetCursor(int x, int y) {
	unsigned char cCmd;

	if (file_i2c < 0 || x < 0 || x > 19 || y < 0 || y > 3)
		return 1;

	switch(y) {
        case 0: 
            cCmd = 0x80; // Адрес первой строки
            break;
        case 1: 
            cCmd = 0xC0; // Адрес второй строки
            break;
        case 2: 
            cCmd = 0x94; // Адрес третьей строки
            break;
        case 3: 
            cCmd = 0xD4; // Адрес четвертой строки
            break;
        default:
            return 1; // Недопустимое значение y
    }

    cCmd += x;
    WriteCommand(cCmd);
	return 0;

}

void lcd2004Shutdown(void) {
	iBackLight = 0;
	WriteCommand(0x08);	
	close(file_i2c);
	file_i2c = -1;
}
