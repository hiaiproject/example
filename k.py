def solution(answers):
    answers = [1,2,3,4,5]
    num1 = [1, 2, 3, 4, 5]
    num2 = [2, 1, 2, 3, 2, 4, 2, 5]
    num3 = [3, 3, 1, 1, 2, 2]
    count = [0, 0, 0]
    answer = []
    for idx in range(len(answers)):
        idx1 = idx % len(num1)
        if (num1[idx1] == answers[idx]):
            count[0] += 1
        idx2 = idx % len(num2)
        if (num2[idx2] == answers[idx]):
            count[1] += 1
        idx3 = idx % len(num3)
        if (num3[idx3] == answers[idx]):
            count[2] += 1
        MAX = max(count)
        print(MAX)
        for idx in range(3):
            if count[idx] == count :
                answer.append(idx+1)
        print(answer)
