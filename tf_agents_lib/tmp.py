"""
for _ in range(2):
    with open('tmp_score_log', 'r') as f:
        arr = list(f.readlines())
        with open('tmp_score_log', 'a') as f:
            if len(arr) > 0:
                f.write(','+str(1))
            else:
                f.write(str(2))

it = 1
for _ in range(2):
    with open('tmp_score_log', 'r') as f:
        arr = f.readlines()
        scores = list(map(int, arr[0].split(',')))
        avg = sum(scores) / len(scores)

    with open('tmp_avg_returns_log', 'a') as f:
        f.write(f"({it},{avg}),")
"""



