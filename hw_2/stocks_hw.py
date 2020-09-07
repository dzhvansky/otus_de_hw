import csv
from pyspark.sql import SparkSession


def pp(s):
    decorator = "=" * 50
    print("\n\n{} {} {}".format(decorator, s, decorator))


if __name__ == "__main__":
    spark = SparkSession.builder.appName("dzhvansky_HW1").getOrCreate()
    sc = spark.sparkContext

    pp("Reading files")
    stocks = sc.textFile('all_stocks_5yr.csv', 100).mapPartitions(lambda x: csv.reader(x))


    def nanFilter(rdd, col_list):
        return rdd.filter(lambda x: all(x[i] != '' for i in col_list))


    def toFloat(rdd, col_list):
        return rdd.map(lambda x: [float(el) if i in col_list else el for i, el in enumerate(x)])


    header = stocks.first()
    stocks = stocks.filter(lambda x: x != header)

    print(header)
    [print(s) for s in stocks.take(3)]

    pp('Check all samples are unique')
    n_rows = stocks.count()
    n_unique_rows = stocks.map(lambda x: (x[0], x[-1])).distinct().count()
    print(n_rows, n_unique_rows)
    assert n_rows == n_unique_rows

    pp("Historical fluctuation")
    openPrice = toFloat(nanFilter(stocks, [1]), [1]).map(lambda x: (x[-1], [x[1], x[0]]))
    minPrice = openPrice.reduceByKey(lambda a, b: a if a[0] < b[0] else b)
    maxPrice = openPrice.reduceByKey(lambda a, b: a if a[0] >= b[0] else b)

    fluctuation = (minPrice.join(maxPrice)
                   .map(lambda x: (x[0], x[1][1][0] - x[1][0][0], x[1][0][1], x[1][1][1]))
                   .sortBy(lambda x: -x[1])
                   )
    # [print(s) for s in fluctuation.take(3)]

    pp("Daily gain")
    closePrice = toFloat(nanFilter(stocks, [4]), [4]).map(lambda x: (x[0], (x[-1], x[4])))
    dateIndex = stocks.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex()
    dateIndexShift = dateIndex.map(lambda x: (x[0], x[1] + 1))


    def joinbyIndex(rdd, index):
        return rdd.join(index).map(lambda x: ((x[1][1], x[1][0][0]), [x[1][0][1], x[0]]))


    dailyGain = (joinbyIndex(closePrice, dateIndex)
                 .join(joinbyIndex(closePrice, dateIndexShift))
                 .map(lambda x: ((x[0][1], x[1][0][1]), x[1][0][0] / x[1][1][0] - 1))
                 .sortBy(lambda x: -x[1])
                 )
    # [print(s) for s in dailyGain.take(3)]

    # def top3reduce(a, b):
    #     va = [v for k, v in a]
    #     vb = [v for k, v in b]
    #     ia = 0
    #     ib = 0
    #     current = 100.
    #     result = []
    #     while ia + ib < 3:
    #         if ((va[ia] >= vb[ib]) & (va[ia] < current)) | (vb[ib] >= current):
    #             result.append(a[ia])
    #             current = va[ia]
    #             ia += 1
    #         else:
    #             result.append(b[ib])
    #             current = vb[ib]
    #             ib += 1
    #     return result

    pp("Price correlation")

    uniqueCompanies = stocks.map(lambda x: (1, x[-1])).distinct()
    uniquePairs = uniqueCompanies.join(uniqueCompanies) \
        .filter(lambda x: x[1][0] != x[1][1]) \
        .map(lambda x: tuple(sorted(x[1]))) \
        .distinct()
    broadcastedPairs = sc.broadcast(uniquePairs.collect())

    n_dates = dateIndex.count()

    meanPrice = toFloat(nanFilter(stocks, [1, 4]), [1, 4]) \
        .map(lambda x: (x[0], (x[-1], (x[1] + x[4]) / 2))) \
        .join(dateIndex) \
        .map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1]))) \
        # .filter(lambda x: x[0] in ['AMZN', 'GOOGL', 'AAL', 'ACN'])

    groupedPrice = meanPrice \
        .groupByKey() \
        .map(lambda x: (x[0], list(x[1]))) \
        .map(lambda x: (x[0], ([i for i, _ in x[1]], [v for _, v in x[1]]))) \
        .map(lambda x: (1, (x[0], x[1])))

    pricePairs = groupedPrice.join(groupedPrice).repartition(4000) \
        .map(lambda x: ((x[1][0][0], x[1][1][0]), (x[1][0][1][0], x[1][1][1][0],
                                                   x[1][0][1][1], x[1][1][1][1]))) \
        .filter(lambda x: x[0] in broadcastedPairs.value) \
        .map(lambda x: (x[0], x[1], [i for i in range(n_dates) if i in x[1][0] and i in x[1][1]])) \
        .filter(lambda x: len(x[2]) != 0) \
        .map(lambda x: (x[0], ([x[1][2][x[1][0].index(i)] for i in x[2]],
                               [x[1][3][x[1][1].index(i)] for i in x[2]]))) \
        .map(lambda x: (x[0], x[1],
                        sum([v1 for v1 in x[1][0]]) / len(x[1][0]),
                        sum([v2 for v2 in x[1][1]]) / len(x[1][1])))\
        .map(lambda x: (x[0], [(v1-x[2], v2-x[3]) for v1, v2 in zip(x[1][0], x[1][1])]))\
        .map(lambda x: (x[0],
                        sum([v1*v2 for v1, v2 in x[1]]) /
                        sum([v1**2 for v1, v2 in x[1]]) ** 0.5 /
                        sum([v2**2 for v1, v2 in x[1]]) ** 0.5))\
        .sortBy(lambda x: -x[1])

        # .map(lambda x: (1, [x, x, x]))\
        # .reduceByKey(lambda a, b: top3reduce(a, b))\

    # # [print(s) for s in pricePairs.take(5)]


    import shutil
    shutil.rmtree('hw1', ignore_errors=True)

    def topNtoWrite(rdd, n, prefix):
        return (rdd
                .zipWithIndex().filter(lambda x: x[1] < n)
                .map(lambda x: (prefix, str(x[0])))
                .groupByKey().map(lambda x: x[0] + ','.join(x[1]).replace(' ', ''))
                )

    result = (topNtoWrite(fluctuation.map(lambda x: x[0]), 3, '2a - ')
               .union(topNtoWrite(dailyGain.map(lambda x: x[0][0]), 3, '2b - '))
               .union(topNtoWrite(pricePairs.map(lambda x: x[0]), 3, '3 - '))
               )


    result.coalesce(1).saveAsTextFile('hw1')

    spark.stop()

    # meanPrice = meanPrice.map(lambda x: (x[1][0], (x[0], x[1][1])))
    # pricePairs = (meanPrice.join(meanPrice)
    #               .map(lambda x: ((x[1][0][0], x[1][1][0]), [x[1][0][1], x[1][1][1], 1]))
    #               .filter(lambda x: x[0][0] != x[0][1])
    #               )
    #
    # pairMeans = (pricePairs
    #              .filter(lambda x: x[0] in broadcastedPairs.value)
    #              .reduceByKey(lambda a, b: [a[0]+b[0], a[1]+b[1], a[2]+b[2]])
    #              .map(lambda x: (x[0], [x[1][0] / x[1][2], x[1][1] / x[1][2]]))
    #              )
    # broadcastedMeans = sc.broadcast(pairMeans.collect())
    #
    # priceCorr = (pricePairs
    #               .filter(lambda x: x[0] in broadcastedPairs.value)
    #               .join(sc.parallelize(broadcastedMeans.value))
    #               .map(lambda x: (x[0], [x[1][0][0]-x[1][1][0], x[1][0][1]-x[1][1][1]]))
    #               .map(lambda x: (x[0], [x[1][0] * x[1][1], x[1][0]**2, x[1][1]**2]))
    #               .reduceByKey(lambda a, b: [a[0]+b[0], a[1]+b[1], a[2]+b[2]])
    #               .map(lambda x: (x[0], x[1][0] / (x[1][1] * x[1][2])**0.5))
    #               .sortBy(lambda x: -x[1])
    # )
    # # [print(s) for s in priceCorr.take(5)]
