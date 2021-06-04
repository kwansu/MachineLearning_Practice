def CreateReport(week):
    reportFileName = "{}주차.txt".format(week)
    reportFile = open(reportFileName, "w", encoding="utf8")
    reportFile.write("- {} 주차 주간보고 -".format(week))
    reportFile.close()

for i in range(1,51):
    CreateReport(i)