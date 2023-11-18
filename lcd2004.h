#ifndef LCD2004_H
#define LCD2004_H

int lcd2004Init(int iChannel, int iAddr);

int lcd2004SetCursor(int x, int y);

int lcd2004Control(int bBacklight, int bCursor, int bBlink);

int lcd2004WriteString(char *szText);

int lcd2004Clear(void);

void lcd2004Shutdown(void);

#endif
