from bs4 import BeautifulSoup
import csv
import sqlite3

if __name__ == '__main__':
	def load_file(file_path, type=""):
		data = []
		with open(file_path, 'r', encoding='latin1') as file_reader:
			reader = csv.reader(file_reader, delimiter='}', quotechar='"')
			next(reader)
			for row in reader:
				if row == []:
					continue

				assert(len(row) == 1)

				row = row[0]
				row = row.split("::")

				if type == "movies":
					genres = row[-1]
					row = row[:-1]
					row.append(genres.replace('|', ','))

				data.append(row)
		return data

	movies = load_file("data/csv/movies.csv", "movies")
	users = load_file("data/csv/users.csv")
	ratings = load_file("data/csv/ratings.csv")

	conn = sqlite3.connect('data/database.db')
	c = conn.cursor()

	# Delete tables if they exist
	c.execute('DROP TABLE IF EXISTS "movies";')
	c.execute('DROP TABLE IF EXISTS "users";')
	c.execute('DROP TABLE IF EXISTS "ratings";')

	#TODO: Create tables in the database and add data to it. REMEMBER TO COMMIT
	c.execute('''CREATE TABLE movies(id TEXT, name TEXT, genres TEXT)''')
	c.execute('''CREATE TABLE users(id TEXT, gender TEXT, age TEXT,
			occupation TEXT, zip TEXT)''')
	c.execute('''CREATE TABLE ratings(user TEXT, movie TEXT, rating TEXT,
			timestamp TEXT)''')

	for movie in movies:
		c.execute('''INSERT INTO movies(id, name, genres)
					  VALUES(?,?,?)''', (movie[0], movie[1], movie[2]))
	for user in users:
		c.execute('''INSERT INTO users(id, gender, age, occupation, zip)
					VALUES(?,?,?,?,?)''', (user[0], user[1], user[2], user[3], user[4]))
	for rating in ratings:
		c.execute('''INSERT INTO ratings(user, movie, rating, timestamp)
					  VALUES(?,?,?,?)''', (rating[0], rating[1], rating[2], rating[3]))
	conn.commit()
