import cmath
import re
def sortalgo():
    name_accs = [('video1area1-frame100-patchsize10%_139x139:', 0.91), ('video1area1-frame100-patchsize5%_221x221:', 1.0), ('video1area1-frame100-patchsize1%_221x221:', 1.0)]
    def getnumber(name_accs,index):
        sum=''
        print(name_accs[index])
        chapter_num = re.findall(r'\d+', name_accs[index][0])
        for b in chapter_num:
            sum=sum+b
        print(sum)
        return sum


    def bubbleSort(name_accs):
        n = len(name_accs)
        arr_list=[]
        for i in range(n):
            arr = getnumber(name_accs, i)
            arr_list.append(int(arr))

        # 遍历所有数组元素
        for i in range(n):
            # Last i elements are already in place
            for j in range(0, n - i - 1):

                if arr_list[j] > arr_list[j + 1]:
                    arr_list[j], arr_list[j + 1] = arr_list[j + 1], arr_list[j]
                    name_accs[j], name_accs[j + 1] = name_accs[j + 1], name_accs[j]

    bubbleSort(name_accs)
    #

    print(name_accs)
    # print("排序后的数组:")
    for i in range(len(name_accs)):
        print(name_accs[i])
class Sort_sort:
    # 排序是个大问题
    def getnumber(self,name_accs,index):
        sum=''
        print(name_accs[index])
        chapter_num = re.findall(r'\d+', name_accs[index][0])
        for b in chapter_num:
            sum=sum+b
        print(sum)
        return sum

    def bubbleSort(self,name_accs):
        n = len(name_accs)
        arr_list=[]
        for i in range(n):
            arr = self.getnumber(name_accs, i)
            arr_list.append(int(arr))

        # 遍历所有数组元素
        for i in range(n):
            # Last i elements are already in place
            for j in range(0, n - i - 1):

                if arr_list[j] > arr_list[j + 1]:
                    arr_list[j], arr_list[j + 1] = arr_list[j + 1], arr_list[j]
                    name_accs[j], name_accs[j + 1] = name_accs[j + 1], name_accs[j]
sort_sort=Sort_sort()
# name_accs = [('video1area1-frame100-patchsize10%_139x139:', 0.91), ('video1area1-frame100-patchsize5%_221x221:', 1.0),
#              ('video1area1-frame100-patchsize1%_221x221:', 1.0)]
# sort_sort.bubbleSort(name_accs)



def dataanalyse():
    with open("a.json","r")as f:
        lines = f.readlines()
    for line in lines:
        if "video1area1-frame50-" in line:
            print(line)



if __name__ == '__main__':
    dataanalyse()