def transformAction(actionBefore, actionBound, actionDim):
    actionAfter = []
    for i in range(actionDim):
        action_i = actionBefore[i]
        actionAfter.append((action_i+1)/2*(actionBound[1] - actionBound[0]) + actionBound[0])
    return actionAfter

