#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <lcd2004.h>
#define MESSAGE1 "people left:"
#define MESSAGE2 "people center:"
#define MESSAGE3 "people right:"
#define MESSAGE4 "total people:"
#define NUM_START_PLACE 17

int main(int argc, char *argv[]) {
	int rc;
	rc = lcd2004Init(1, 0x27);
	if (rc)
	{
		printf("Initialization failed; aborting...\n");
		return 0;
	}

	int left, center, right;
	char buf1[21];
	char buf2[21];
	char buf3[21];
	char buf4[21];

	buf1[20] = '\0';
	buf2[20] = '\0';
	buf3[20] = '\0';
	buf4[20] = '\0';

	memset(buf1, ' ', sizeof(buf1));
	memset(buf2, ' ', sizeof(buf2));
	memset(buf3, ' ', sizeof(buf3));
	memset(buf4, ' ', sizeof(buf4));
	
    memcpy(buf1, MESSAGE1, strlen(MESSAGE1));
    memcpy(buf2, MESSAGE2, strlen(MESSAGE2));
    memcpy(buf3, MESSAGE3, strlen(MESSAGE3));
	memcpy(buf4, MESSAGE4, strlen(MESSAGE4));

	lcd2004SetCursor(0,0);
	lcd2004WriteString(buf1);

	lcd2004SetCursor(0,1);
	lcd2004WriteString(buf2);
	lcd2004SetCursor(0,2);
	lcd2004WriteString(buf3);


	lcd2004SetCursor(0,3);
	lcd2004WriteString(buf4);

    while (1) {
        if (scanf("%d %d %d", &left, &center, &right) > 0) {
			snprintf(buf1 + NUM_START_PLACE, sizeof(buf1) - NUM_START_PLACE, "%d", left);
			snprintf(buf2 + NUM_START_PLACE, sizeof(buf2) - NUM_START_PLACE, "%d", center);
			snprintf(buf3 + NUM_START_PLACE, sizeof(buf3) - NUM_START_PLACE, "%d", right);
			snprintf(buf4 + NUM_START_PLACE, sizeof(buf4) - NUM_START_PLACE, "%d", left+center+right);


			printf("%s\n", buf1);
			printf("%s\n", buf2);
			printf("%s\n", buf3);
			printf("%s\n", buf4);

			lcd2004SetCursor(NUM_START_PLACE, 0);
			lcd2004WriteString(buf1 + NUM_START_PLACE);

			lcd2004SetCursor(NUM_START_PLACE, 1);
			lcd2004WriteString(buf2 + NUM_START_PLACE);
			lcd2004SetCursor(NUM_START_PLACE, 2);
			lcd2004WriteString(buf3 + NUM_START_PLACE);


			lcd2004SetCursor(NUM_START_PLACE, 3);
			lcd2004WriteString(buf4 + NUM_START_PLACE);
        }
    }

	lcd2004Shutdown();
	return 0;
}
