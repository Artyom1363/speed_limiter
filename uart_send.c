#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>

// gcc -o uart_send uart_send.c
int main() {
    int serial_port = open("/dev/ttyS0", O_WRONLY);

    if (serial_port < 0) {
        printf("Ошибка %i открытия: %s\n", errno, strerror(errno));
        return 1;
    }

    // Настройка параметров UART
    struct termios tty;
    if(tcgetattr(serial_port, &tty) != 0) {
        printf("Ошибка %i от tcgetattr: %s\n", errno, strerror(errno));
        return 1;
    }

    tty.c_cflag &= ~PARENB; // Отключение бита четности
    tty.c_cflag &= ~CSTOPB; // Один стоп-бит
    tty.c_cflag &= ~CSIZE; // Очистка битов размера
    tty.c_cflag |= CS8; // 8 битов на байт
    tty.c_cflag &= ~CRTSCTS; // Отключение аппаратного управления потоком
    tty.c_cflag |= CREAD | CLOCAL; // Включение чтения; игнорировать управляющие линии

    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO; // Отключение эхо
    tty.c_lflag &= ~ECHOE; // Отключение эхо стирания
    tty.c_lflag &= ~ECHONL; // Отключение эхо новой строки
    tty.c_lflag &= ~ISIG; // Отключение интерпретации INTR, QUIT, SUSP

    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // Отключение программного управления потоком
    tty.c_iflag &= ~(IGNBRK|BRKINT|ISTRIP|INLCR|IGNCR|ICRNL); // Отключение специальных обработок символов

    tty.c_oflag &= ~OPOST; // Отключение специальных обработок вывода
    tty.c_oflag &= ~ONLCR; // Отключение преобразования новой строки

    // Настройка скорости передачи данных
    cfsetispeed(&tty, B9600);
    cfsetospeed(&tty, B9600);

    // Сохранение настроек tty
    if (tcsetattr(serial_port, TCSANOW, &tty) != 0) {
        printf("Ошибка %i от tcsetattr: %s\n", errno, strerror(errno));
        return 1;
    }

    // Отправка строки через UART
    int num;
    char buf[3];
    while (1) {
        // Чтение числа из stdin
        if (scanf("%d", &num) > 0) {
            sprintf(buf, "%d", num);
            buf[2] = '\n';
            write(serial_port, buf, sizeof(buf));
            // printf("C Программа получила число: %d\n", num);
            fflush(stdout); // Очистка буфера stdout для немедленного вывода
        }
    }
    // char msg[] = "Привет через UART!\n";
    

    // Закрытие UART порта
    close(serial_port);
    return 0;
}

