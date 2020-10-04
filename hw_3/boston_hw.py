from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F


def pp(s):
    decorator = "=" * 50
    print("\n\n{} {} {}".format(decorator, s, decorator))


if __name__ == "__main__":

    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()

    pp('dataset')
    crimeFacts = spark.read\
        .option("header", "true")\
        .option("inferSchema", "true")\
        .csv("boston_crimes/crime.csv").fillna({'DISTRICT': 'NaN'})
    crimeFacts.show(truncate=False)
    crimeFacts.printSchema()

    offenseCodes = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("boston_crimes/offense_codes.csv")
    offenseCodes.show()

    print('N rows = ', offenseCodes.count())
    print('N unique CODEs = ', offenseCodes.select('CODE').distinct().count())
    print('N rows without duplicates = ', offenseCodes.dropDuplicates().count())
    # offenseCodes.dropDuplicates().groupBy('CODE').count().orderBy('count', ascending=False).show()
    print('N rows without duplicates by CODE = ', offenseCodes.dropDuplicates(['CODE']).count())
    offenseCodes = offenseCodes.dropDuplicates(['CODE'])


    pp('crimes_total')
    crimesTotal = crimeFacts\
        .groupBy('DISTRICT')\
        .count()\
        .alias('crimes_total')

    crimesTotal.show(truncate=False)

    crimeFacts.filter(F.col('Lat')<42.).show(truncate=False)
    crimeFacts.filter(F.col('Long') > -70.).show(truncate=False)


    pp('crimes_monthly')
    crimesMonthly = crimeFacts\
        .groupBy('DISTRICT', 'YEAR', 'MONTH')\
        .count()\
        .groupBy('DISTRICT')\
        .agg(F.expr('percentile_approx(count, 0.5)').alias('crimes_monthly'))

    crimesMonthly.show(truncate=False)


    pp('frequent_crime_types')
    windowSpec = Window.partitionBy('DISTRICT').orderBy(F.col('count').desc())

    frequentCrimeTypes = crimeFacts\
        .join(F.broadcast(offenseCodes), offenseCodes.CODE == crimeFacts.OFFENSE_CODE)\
        .select('DISTRICT', F.split(offenseCodes.NAME, ' - ').getItem(0).alias('crime_type')) \
        .groupBy('DISTRICT', 'crime_type')\
        .count()\
        .orderBy('DISTRICT', 'count', ascending=False)\
        .withColumn('rank', F.rank().over(windowSpec))\
        .filter(F.col('rank') <= 3)\
        .groupBy('DISTRICT')\
        .agg(F.concat_ws(', ', F.collect_list(F.col('crime_type'))).alias('frequent_crime_types'))

    frequentCrimeTypes.show(truncate=False)


    pp('lat')
    latCrime = crimeFacts\
        .filter(F.col('Lat') != -1.0)\
        .groupBy('DISTRICT')\
        .agg(F.mean('Lat').alias('lat'))

    latCrime.show(truncate=False)


    pp('lng')
    lngCrime = crimeFacts\
        .filter(F.col('Long') != -1.0)\
        .groupBy('DISTRICT') \
        .agg(F.mean('Long').alias('lng'))

    lngCrime.show(truncate=False)


    pp('results')
    result = crimesTotal\
        .join(crimesMonthly, on='DISTRICT', how='left')\
        .join(frequentCrimeTypes, on='DISTRICT', how='left')\
        .join(latCrime, on='DISTRICT', how='left')\
        .join(lngCrime, on='DISTRICT', how='left')

    result.show(truncate=False)
