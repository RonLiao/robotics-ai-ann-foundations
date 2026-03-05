import serial
import time
import argparse

def calculate_checksum(packet):
    # Checksum = ~(ID + Length + Instruction + Parameter1 + ... + ParameterN) & 0xFF
    return (~sum(packet[2:-1])) & 0xFF

def send_packet(ser, servo_id, instruction, parameters):
    length = len(parameters) + 2
    packet = [0xFF, 0xFF, servo_id, length, instruction] + parameters + [0]
    packet[-1] = calculate_checksum(packet)
    
    # print(f"Sending packet: {[hex(b) for b in packet]}")
    ser.write(bytearray(packet))
    time.sleep(0.1)
    
    # Read response (optional)
    if ser.in_waiting:
        response = ser.read(ser.in_waiting)
        # print(f"Response: {[hex(b) for b in response]}")
    else:
        print(f"No response from motor ID {servo_id}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced: Force Feetech STS3215 motor to Position Mode (0) and clear offsets.')
    parser.add_argument('--port', type=str, required=True, help='Serial port (e.g., /dev/ttyACM0)')
    parser.add_argument('--id', type=int, default=5, help='Motor ID (default: 5)')
    
    args = parser.parse_args()
    
    try:
        ser = serial.Serial(args.port, 1000000, timeout=1)
        print(f"Opened port {args.port} at 1M baud.")
        
        # 0. Disable Torque (Address 40 = 0)
        print(f"Step 0: Disabling Torque for ID {args.id}...")
        send_packet(ser, args.id, 0x03, [40, 0])

        # 1. Unlock EPROM (Address 55 = 0)
        print(f"Step 1: Unlocking EPROM...")
        send_packet(ser, args.id, 0x03, [55, 0])
        
        # 2. Reset Homing Offset to 0 (Address 20, Length 2)
        print(f"Step 2: Clearing Homing Offset (Address 20)...")
        send_packet(ser, args.id, 0x03, [20, 0, 0])
        
        # 3. Set Operating Mode to Position Mode (Address 33 = 0)
        print(f"Step 3: Setting Operating Mode to 0 (Position Mode)...")
        send_packet(ser, args.id, 0x03, [33, 0])
        
        # 4. Lock EPROM (Address 55 = 1)
        print(f"Step 4: Locking EPROM...")
        send_packet(ser, args.id, 0x03, [55, 1])
        
        # 5. Enable Torque (Address 40 = 1)
        print(f"Step 5: Enabling Torque...")
        send_packet(ser, args.id, 0x03, [40, 1])

        print("\n" + "="*50)
        print("SUCCESS! DATA WRITTEN TO EPROM.")
        print("CRITICAL: YOU MUST POWER CYCLE THE MOTOR NOW!")
        print("Unplug the power cable, wait 5 seconds, and plug it back in.")
        print("="*50 + "\n")
        
        ser.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
