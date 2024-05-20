import website
import pyodbc
import datetime

server = ''
database = ''
username = ''
password = ''
driver = '{ODBC Driver 18 for SQL Server}'  # This may vary depending on your environment

# Establish connection
conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}')


        
     



def admin(email, username, password):
    cursor = conn.cursor()
    role = 'user'

    # Insert data into the users table
    cursor.execute("INSERT INTO users (email, username, password, role) VALUES (?, ?, ?, ?)", (email, username, password, role))
    conn.commit()

def authenticate_user(email, password):
    cursor = conn.cursor()
    cursor.execute("SELECT email, role FROM users WHERE email=? AND password=?", (email, password))
    user = cursor.fetchone()
    if user:
        # Update the last_login timestamp in the database
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("UPDATE users SET last_login=? WHERE email=?", (current_time, email))
        conn.commit()
    return user

def users():
    cursor = conn.cursor()
    cursor.execute("SELECT email, username, password, last_login FROM users")
    users = cursor.fetchall()
    return users

def UserComponents(user_id):
    cursor = conn.cursor()
    select_user_components_query = '''
    SELECT component, value, date
    FROM user_components, users
    WHERE users.email=user_components.user_id AND
    users.email= ?
    '''
    cursor.execute(select_user_components_query, (user_id,))
    components = cursor.fetchall()
    return components 


def insert_data():
    # Assuming you have these functions defined in your 'website' module
    csv_files = website.datafile.csv_files()
    dataset_dir = "datasets"

    website.datafile.configure_dataset_directory(csv_files, dataset_dir)
    df = website.datafile.create_df(dataset_dir, csv_files)

    cursor = conn.cursor()

    for file, dataframe in df.items():
        tbl_name = website.datafile.clean_tbl_name(file)
        col_str, dataframe_columns = website.datafile.clean_colname(dataframe)

        # Create the table if it doesn't exist
        create_table_query = f'''
            CREATE TABLE IF NOT EXISTS {tbl_name} (
                {col_str}
            )
        '''
        cursor.execute(create_table_query)
        conn.commit()

        # Insert data into the Azure SQL Database table
        for index, row in dataframe.iterrows():
            insert_query = f'''
                INSERT INTO {tbl_name} ({','.join(dataframe_columns)})
                VALUES ({','.join(['?']*len(dataframe_columns))})
            '''
            cursor.execute(insert_query, tuple(row))
            conn.commit()

    cursor.close()