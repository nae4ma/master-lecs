## 問4: 与えられたリストを逆順にして返す
def reverse(xlist):
  xlist_reverse = []
  for i in range(1, len(xlist)+1):
    xlist_reverse.append(xlist[len(xlist) - i])

  return xlist_reverse


xlist = [10, 20, 30, 40, 50]
print(reverse(xlist))

xlist = ["cat", "dog", "lion", "tiger"]
print(reverse(xlist))