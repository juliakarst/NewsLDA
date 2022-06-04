import csv
import glob

"This python-file will merge multiple smaller csv-files to one big one"
header = []
data = []

file_list = glob.glob('/Users/juliakarst/PycharmProjects/NewsWebcrawler/csv_data_and_scripts/big_csv_files/*.csv')


def merge_data(input_files):
    for file in input_files:
        with open(file, 'r') as readingfile:
            reader = csv.reader(readingfile, delimiter=',')
            header = next(reader)
            data.extend([row for row in reader])


merge_data(file_list)

with open('all_articles_new.csv', 'a') as result:
    writer = csv.writer(result, delimiter=',')
    writer.writerow(["title", "author", "date_published", "url", "article_text", "source"])
    writer.writerows(data)