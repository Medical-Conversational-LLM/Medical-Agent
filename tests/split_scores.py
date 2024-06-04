scores = ['', '1: 0.9', '2: 0.7', '3: 0.8', '4: 0.5', '5: 0.3', '6: 0.7', '7: 0.1', '8: 0.6', '9: 0.8', '10: 0.4',
          '11: 0.9', '12: 0.7', '13: 0.8', '14: 0.9', '15: 0.6', '16: 0.7', '17: 0.6', '18: 0.8', '19: 0.1', '20: 0.6', '21: 0.8']



def handle_scores(scores:list[str]):
    scores = [score.split(":") for score in scores if score != ""]
    scores = [[int(score[0]), float(score[1]) ] for score in scores if score != ""]

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return scores
    

print(handle_scores(scores))


