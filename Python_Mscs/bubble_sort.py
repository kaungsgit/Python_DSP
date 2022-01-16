def bubble_sort(a):
    for j in range(a.__len__() - 1):
        for i in range(a.__len__() - 1):
            if a[i] >= a[i + 1]:
                t = a[i + 1]
                a[i + 1] = a[i]
                a[i] = t


    # for j in range(a.__len__() - 1):
    #     for i in range(j + 1, a.__len__() - 1):
    #         val1 = a[j]
    #         val2 = a[i]
    #
    #         if val2 < val1:
    #             a[j] = val2
    #             a[j + 1] = val1


abc = [1, 4, 2, 3, 8, 2]

bubble_sort(abc)
print(abc)
