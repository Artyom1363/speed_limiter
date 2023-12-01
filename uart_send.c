#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// определение базового адреса переферийных устройств в RPI 4b
#define BCM2711_PERI_BASE       0xFE000000
#define GPIO_BASE               (BCM2711_PERI_BASE + 0x200000)  // базовый адрес GPIO контроллера
#define UART_BASE               (BCM2711_PERI_BASE + 0x201000) // базовый адрес UART

#define BLOCK_SIZE              (4 * 1024)


#define GPPUD				*(gpio.addr + 0x94/4)
#define GPPUDCLK0			*(gpio.addr + 0x98/4)
#define UART_DR				*(uart.addr + 0x00/4)
#define UART_RSRECR			*(uart.addr + 0x04/4)
#define UART_FR				*(uart.addr + 0x18/4)
#define UART_ILPR			*(uart.addr + 0x20/4)
#define UART_IBRD			*(uart.addr + 0x24/4)
#define UART_FBRD			*(uart.addr + 0x28/4)
#define UART_LCRH			*(uart.addr + 0x2C/4)
#define UART_CR				*(uart.addr + 0x30/4)
#define UART_IFLS			*(uart.addr + 0x34/4)
#define UART_IMSC			*(uart.addr + 0x38/4)
#define UART_RIS			*(uart.addr + 0x3C/4)
#define UART_MIS			*(uart.addr + 0x40/4)
#define UART_ICR			*(uart.addr + 0x44/4)
#define UART_DMACR			*(uart.addr + 0x48/4)
#define UART_ITCR			*(uart.addr + 0x80/4)
#define UART_ITIP			*(uart.addr + 0x84/4)
#define UART_ITOP			*(uart.addr + 0x88/4)
#define UART_TDR			*(uart.addr + 0x8c/4)

struct bcm2835_peripheral {
    unsigned long addr_p;
    int mem_fd;
    void *map;
    volatile unsigned int *addr;
};

struct bcm2835_peripheral gpio = { GPIO_BASE };
struct bcm2835_peripheral uart = { UART_BASE };


// отображает физический адрес в адресное пространство процесса
int map_peripheral(struct bcm2835_peripheral *p) {

    if ((p->mem_fd = open("/dev/mem", O_RDWR | O_SYNC) ) < 0) {
        fprintf(stderr, "Failed to open /dev/mem, try checking permissions.");
        return -1;
    }

    p->map = mmap(
                NULL,
                BLOCK_SIZE,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                p->mem_fd,      // файловый дескриптор
                p->addr_p       // отображаем физический адрес памяти
                );

    if (p->map == MAP_FAILED) {
        fprintf(stderr, "mmap faild\n");
        return -1;
    }
    p->addr = (volatile unsigned int *)p->map;
    return 0;
}

void unmap_peripheral(struct bcm2835_peripheral *p) {
    munmap(p->map, BLOCK_SIZE);
    close(p->mem_fd);
}

static inline void delay(int32_t count) {
    asm volatile("__delay_%=: subs %[count], %[count], #1; bne __delay_%=\n"
                 : "=r"(count)
                 : [count] "0"(count)
                 : "cc");
}


void uart_init() {
	// отключает подтягивающие резисторы
	GPPUD = 0x00000000;
	delay(150);

    // установка поддтягивающих резисторов для 14, 15 пинов
	GPPUDCLK0 = (1 << 14) | (1 << 15);
	delay(150);
	// запись 0, чтобы настройки применились
	GPPUDCLK0 =  0x00000000;

	// обнуление регистра управления
	UART_CR = 0x00000000;

	// очистка флагового регистра
	UART_FR = 0X00000000;

	// очистка ожидающих прерываний
	UART_ICR =  0x7FF;
	
	
    // Установка целой части делителя для определения скорости передачи данных
    // 48000000 / (16 * 9600) = 312.5 = 312
	UART_IBRD = 312;

	// дробная часть делителя  (0.5 * 64) + 0.5 = 32
	UART_FBRD = 32;

    // fifo - временный буфер
	// очистка бита fifo
	UART_LCRH = (0 << 4);

	// Установка 8-битного формата данных (5, 6), режим FIFO (4) и 1 стоп-бит без проверки четности.
	UART_LCRH = (1 << 4) | (1 << 5) | (1 << 6);

	// маскирование прерываний
	UART_IMSC = (1 << 1) | (1 << 4) | (1 << 5) | (1 << 6) |
                               (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10);

    // разрешение приема и передачи данных                  
	UART_CR = (1 << 0) | (1 << 8) | (1 << 9);
}


void uart_putc(unsigned char c) {
    while (UART_FR & (1 << 5)); // пока не освободится место в буфере fifo, 5ый бит отвечает за заполненность
    UART_DR = (unsigned char) c; // запись в регистр данных
}


void uart_puts(const char *str) {
    for (size_t i = 0; str[i] != '\0'; i++)
        uart_putc((unsigned char)str[i]);
}

int main() {

    if(map_peripheral(&gpio) == -1) {
        fprintf(stderr, "Failed\n");
        return -1;
    }

    if(map_peripheral(&uart) == -1) {
        fprintf(stderr, "Failed\n");
        return -1;
    }

	unsigned char c;	
	uart_init();
	
	// режим без буферизации
	setvbuf(stdout, NULL, _IONBF, 0);

	int num;
    int BUF_LEN = 4;
    char buf[BUF_LEN];
    buf[2] = '\n';
    buf[3] = '\0';
    while (1) {
        // Чтение числа из stdin
        if (scanf("%d", &num) > 0) {
            sprintf(buf, "%d", num);
            // for (int i = 0; i < BUF_LEN; ++i)
            uart_puts(buf);
        }
    }
	unmap_peripheral(&gpio);
	unmap_peripheral(&uart);
    return 0;
}