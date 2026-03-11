# Example 3.11 for C_3^2, first prime is q = 960643, may take few seconds
ps = [2, 3, 5, 7]
found = False
for q in Primes():
    if q % 9 == 1:
        for p in ps:
            if power_mod(p, (q - 1) // 9, q) != 1:
                break
            if p == ps[-1]:
                print(q)
                found = True
                break
    if found:
        break

# Example 3.30 for C_2^3, first prime is q = 14836487689, make take an hour
ps = [2, 3, 5, 7, 11, 17, 23, 29, 31]
found = False
for q in Primes():
    if q % 8 == 1:
        for p in ps:
            if power_mod(p, (q - 1) // 8, q) != 1:
                break
            if p == ps[-1]:
                print(q)
                found = True
                break
    if found:
        break
