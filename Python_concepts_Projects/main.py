# Dictionary for Morse Code translation
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..', '1': '.----', '2': '..---', '3': '...--', '4': '....-', 
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.', 
    '0': '-----', ',': '--..--', '.': '.-.-.-', '?': '..--..', 
    "'": '.----.', '-': '-....-', '/': '-..-.', '(': '-.--.', 
    ')': '-.--.-', '&': '.-...', ':': '---...', ';': '-.-.-.', 
    '=': '-...-', '+': '.-.-.', '_': '..--.-', '"': '.-..-.', 
    '$': '...-..-', '!': '-.-.--', '@': '.--.-.', ' ': '/'
}

def string_to_morse(input_text):
    """Convert a string to Morse code."""
    # Convert to uppercase for consistency
    input_text = input_text.upper()
    
    # Translate each character to Morse code
    morse_code = []
    for char in input_text:
        if char in MORSE_CODE_DICT:
            morse_code.append(MORSE_CODE_DICT[char])
        else:
            morse_code.append('?')  # Placeholder for unsupported characters
    
    # Join Morse code with a space
    return ' '.join(morse_code)

def main():
    print("Welcome to the Morse Code Converter!")
    user_input = input("Enter a string to convert to Morse Code: ")
    morse_result = string_to_morse(user_input)
    print(f"Morse Code:\n{morse_result}")

if __name__ == "__main__":
    main()
