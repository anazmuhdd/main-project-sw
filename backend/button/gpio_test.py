from periphery import GPIO
import time

GPIO_CHIP = "/dev/gpiochip0"
GPIO_LINE = 108   # PD12

button = GPIO(GPIO_CHIP, GPIO_LINE, "in")

while True:
    val = button.read()
    print("val:",val) 
    print("Pressed" if val != 0 else "Released")
    time.sleep(0.2)
