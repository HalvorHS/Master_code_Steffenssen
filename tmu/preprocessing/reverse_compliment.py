def reverse_complement(dna_strings):
    complements = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reverse_complements = []

    for dna_string in dna_strings:
        # Reverse the string and convert to uppercase
        reversed_string = dna_string[::-1].upper()

        # Replace each nucleotide with its complement, keeping other characters unchanged
        complemented_string = ''.join(complements.get(base, base) for base in reversed_string)

        reverse_complements.append(complemented_string)

    return reverse_complements

def reverse_strings(strings):
    reversed_strings = []

    for string in strings:
        # Reverse the string and convert to uppercase
        reversed_string = string[::-1]

        reversed_strings.append(reversed_string)

    return reversed_strings