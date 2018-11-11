#include <stdio.h>
const char name = 'n';
const int date = 20181111;
int test_int1, test_int2, test_int3;
void _print() {
    printf("hello world");
}
int fibonacci(int n) {
    if (n == 0)
        return (0);
    if (n == 1)
        return (1);
    if (n >= 2)
        return (fibonacci(n-1) + fibonacci(n-2));
}
int main() {
    const int n = 10;
    const char c1 = 'a';
    int fibonacci_ans;
    char test_c, test_d, c;
    int i, sum;
    char array[10];
    i = 0;
    sum = 0;
    scanf("%c", &test_c);
    fibonacci_ans = fibonacci(n);
    printf("fibonacci_ans:%d", fibonacci_ans);
    switch (test_c) {
        case 'a':c = 'd';
        case 'b':c = 'c';
        case 'c':c = 'b';
        default:c = 'x';
    }
    printf("%c", c);
    while (i < 100) {
        sum = sum + i;
        i = i + 1;
    }
    printf("    ");
    printf("sum of 100:%d", sum);
    printf("    ");
    _print();
    printf("    ");
    i = 0;
    test_int1 = 10;
    test_d = 'n';
    while (i < test_int1) {
        array[i] = test_d;
        printf("%c", array[i]);
        i = i + 1;
    }
    printf("    ");
    printf("end");
    return 0;
}
