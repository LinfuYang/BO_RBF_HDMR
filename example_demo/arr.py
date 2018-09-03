import numpy as np
x = [[0, 1], [0, 2], [1, 2], [2, 4], [3, 5], [6, 7]]
y = [[0, 1], [0, 2], [1, 4], [2, 3]]

def is_xiangguan_2D(second_order=None):
    '''
    # second_order用来存放变量存在共项元素的关系
    # 例如：denpend_func_2: [[0, 1, 2], [1, 2], [2]] 表示 second_order中第一项与第二项、第三项共享元素，第一项和第二项共享元素
    denpend_func_2 = []
    for i in range(len(second_order)):
        temp_arr = [i]
        if len(second_order) > 0:
            for j in range(len(second_order)):
                if i < j:
                    if ((second_order[i][0] == second_order[j][0]) or (second_order[i][0] == second_order[j][1])) or \
                       ((second_order[i][1] == second_order[j][0]) or (second_order[i][1] == second_order[j][1])):

                        temp_arr.append(j)

            denpend_func_2.append(temp_arr)
    print('denpend_func_2:', denpend_func_2)

    # 二阶函数中存在共享元素的函数的 在x_inter中的下标
    denpend_d = []
    i = 0
    while i < len(denpend_func_2):
        # print('i:', i)
        temp_arr = denpend_func_2[i]
        i = i + 1
        for idex in range(i, len(denpend_func_2)):
            tpm = denpend_func_2[idex][0]
            # print('idex:', idex)
            if tpm in temp_arr:
                temp_arr.extend(denpend_func_2[idex])
                i = idex + 1
            else:
                break
        denpend_d.append(list(np.unique(temp_arr)))

    # print('denpend_d:', denpend_d)

    # 判断一维和二维

    由于denpend_d表示的是二维函数间是否存在共享变量

    depend_2D = []
    independ_2D = []
    # 不存一阶函数的情况
    for temp in denpend_d:
        if len(temp) > 1:
            depend_2D.append(temp)
        else:
            independ_2D.append(temp)

    return depend_2D, independ_2D
    '''
    # print(second_order)
    xishu = []

    i = 0

    while i < len(second_order):
        index_x = []
        for inx in xishu:
            index_x.extend(inx)
        index_x = list(np.unique(index_x))
        # print('index_x:', index_x)
        while i < len(index_x):
            if i in index_x:
                i += 1
            else:
                break

        if i == len(second_order):
            break
        else:
            # print(i)
            temp = second_order[i].copy()
            index = [i]
            j = 0
            while j < len(second_order):
                if j not in index:
                    if (second_order[j][0] in temp) or (second_order[j][1] in temp):
                        index.append(j)
                        temp.extend(second_order[j])
                        temp = list(np.unique(temp))
                        index = list(np.unique(index))
                        j = 0
                    else:
                        j += 1
                else:
                    j = j + 1
            # print(temp)
            # print(index)
            xishu.append(index)
            i += 1

    return xishu

xishu = is_xiangguan_2D(second_order=x)
print(xishu)

# epend_2D, independ_2D = is_xiangguan_2D(second_order=x)
# rint('depend_2D:', depend_2D)
# rint('independ_2D:', independ_2D)