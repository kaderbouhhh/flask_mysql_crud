from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flaskext.mysql import MySQL
import csv
from werkzeug.utils import secure_filename
import os
import pandas as pd
import io
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

app = Flask(__name__)

# Replace with your MySQL configuration
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'crud_db'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'

mysql = MySQL(app)

app.secret_key = os.urandom(24)

# Ensure the 'uploads' directory exists within 'static'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Home route to display all records
@app.route('/')
def index():
    conn = mysql.connect()
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, author, cover_image FROM books')
    books = cursor.fetchall()
    cursor.close()
    conn.close()
    # Convert query result to a list of dictionaries
    books = [{'id': book[0], 'title': book[1], 'author': book[2], 'cover_image': book[3]} for book in books]
    return render_template('index.html', books=books)

# Other routes here...

@app.route('/classification')
def classification():
    return render_template('classification.html')

@app.route('/clustering')
def clustering():
    return render_template('clustering.html')

@app.route('/import_csv', methods=['POST'])
def import_csv():
    if 'csvFile' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('index'))

    file = request.files['csvFile']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('index'))

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        conn = mysql.connect()
        cursor = conn.cursor()
        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cursor.execute('INSERT INTO books (title, author, cover_image) VALUES (%s, %s, %s)',
                               (row['Title'], row['Author'], row.get('CoverImage', 'default.png')))
        conn.commit()
        cursor.close()
        conn.close()
        flash('Books imported successfully!', 'success')
    else:
        flash('Invalid file format. Please upload a CSV file.', 'danger')

    return redirect(url_for('index'))

# Route to add a new book
@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        title = request.form.get('title')
        author = request.form.get('author')

        # Handle the file upload
        cover_image = request.files.get('cover_image')
        if cover_image and cover_image.filename != '':
            filename = secure_filename(cover_image.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            cover_image.save(filepath)
            filepath = filepath.replace('static/', '')  # Store relative path to the image
        else:
            filepath = 'uploads/default.png'  # Use a default image if none is uploaded

        conn = mysql.connect()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO books (title, author, cover_image) VALUES (%s, %s, %s)',
                       (title, author, filepath))
        conn.commit()
        cursor.close()
        conn.close()
        flash('Book added successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('add.html')

@app.route('/update/<int:id>', methods=['GET', 'POST'])
def update(id):
    conn = mysql.connect()
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, author, cover_image FROM books WHERE id = %s', (id,))
    book_data = cursor.fetchone()
    if not book_data:
        flash('Book not found!', 'danger')
        return redirect(url_for('index'))

    book = {
        'id': book_data[0],
        'title': book_data[1],
        'author': book_data[2],
        'cover_image': book_data[3]
    }

    if request.method == 'POST':
        title = request.form.get('title')
        author = request.form.get('author')
        cover_image = request.files.get('cover_image')
        
        if cover_image and cover_image.filename != '':
            filename = secure_filename(cover_image.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            cover_image.save(filepath)
            filepath = filepath.replace('static/', '')
        else:
            filepath = book['cover_image']

        cursor.execute('UPDATE books SET title = %s, author = %s, cover_image = %s WHERE id = %s',
                       (title, author, filepath, id))
        conn.commit()
        flash('Book updated successfully!', 'success')
        return redirect(url_for('index'))

    cursor.close()
    conn.close()
    return render_template('update.html', book=book)



@app.route('/delete/<int:id>')
def delete(id):
    conn = mysql.connect()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM books WHERE id = %s', (id,))
    conn.commit()
    cursor.close()
    conn.close()
    flash('Book deleted successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/regression')
def regression():
    return render_template('regression.html')

@app.route('/upload_and_process', methods=['POST'])
def upload_and_process():
    global df
    if 'dataset' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['dataset']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        return jsonify({'columns': df.columns.tolist()}), 200
    else:
        return jsonify({'error': 'Invalid file type, please upload a CSV file'}), 400

@app.route('/visualize_data', methods=['POST'])
def visualize_data():
    data = request.get_json()
    print("Received Data:", data)  # Print received data for debugging
    features = data['features']
    label = data.get('label')
    plot_types = data.get('plot_types', ['scatter'])  # Default to scatter plots

    # Check if df is populated
    if df.empty:
        return jsonify({'error': 'No dataset loaded'}), 400

    # Create specified plots for each feature
    graphs = []
    for feature, plot_type in zip(features, plot_types):
        plt.figure()
        
        if plot_type == 'scatter':
            plt.scatter(df[feature], df[label])
            plt.xlabel(feature)
            plt.ylabel(label)
            plt.title(f'{feature} vs {label}')
        
        elif plot_type == 'histogram':
            plt.hist(df[feature], bins=10, alpha=0.7)
            plt.xlabel(feature)
            plt.title(f'Histogram of {feature}')
        
        elif plot_type == 'boxplot':
            sns.boxplot(x=df[feature])
            plt.title(f'Boxplot of {feature}')
        
        elif plot_type == 'heatmap':
            if len(features) > 1:
                sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
                plt.title('Correlation Heatmap')
            else:
                return jsonify({'error': 'Heatmap requires at least two features'}), 400
        
        # Save plot to a string in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph = base64.b64encode(buf.read()).decode('utf-8')
        graphs.append(graph)
        buf.close()
        plt.close()  # Close the figure to avoid memory issues

    return jsonify({'graphs': graphs})

@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.get_json()
    features = data['features']
    label = data['label']

    # Extract the relevant columns
    X = df[features]
    y = df[label]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return jsonify({'model': 'Linear Regression', 'r2_score': r2, 'mse': mse})

@app.route('/make_prediction', methods=['POST'])
def make_prediction():
    # Your prediction logic here
    pass


@app.route('/upload_and_process_classification', methods=['POST'])
def upload_and_process_classification():
    if 'dataset' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['dataset']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        try:
            # Read the CSV file
            df = pd.read_csv(filename)
            
            # Get column names
            columns = df.columns.tolist()
            
            # Example: Return the column names for feature and label selection
            return jsonify({'columns': columns})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400
@app.route('/visualize_classification_data', methods=['POST'])
def visualize_classification_data():
    # Process the request data and return a response
    # Example: perform classification and generate visualizations
    return jsonify({
        'graphs': []  # Replace with actual graph data
    })


if __name__ == '__main__':
    global df
    df = None  # To store the uploaded dataframe
    app.run(debug=True)

