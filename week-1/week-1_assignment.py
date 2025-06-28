n = int(input("Enter the number of rows: "))

print("\nLower Triangle:")
for i in range(1, n + 1):
    for j in range(i):
        print("*", end=" ")
    print()

print("\nUpper Triangle:")
for i in range(n, 0, -1):
    for j in range(i):
        print("*", end=" ")
    print()

print("\nPyramid:")
for i in range(1, n + 1):
    for space in range(n - i):
        print(" ", end="")
    for star in range(i):
        print("* ", end="")
    print()
