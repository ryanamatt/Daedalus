import pytest
import random
from daedalus import Matrix, DataFrame

def test_dataframe_initialization():
    """Verify DataFrame initializaes correctly"""
    df1 = DataFrame()

    assert df1

    df2 = DataFrame('Test', [1])
    assert df2
    assert df2.rows == 1
    assert df2.cols == 1
    assert df2.get_column_names() == ['Test']

    with pytest.raises(TypeError):
        DataFrame('Test', 1)

    with pytest.raises(TypeError):
        DataFrame(1, [1])

def test_dataframe_repr():
    """Verify __repr__ of DataFrame"""
    df1 = DataFrame('Test', [1, 2])
    df1.add_column('Test2', [3, 4])
    df1_cols = df1.get_column_names()
    dash = "-" * 8 * df1.cols

    df1_string = (
        f"DataFrame ({df1.rows} rows x {df1.cols} cols)\n"
        f"{df1_cols[0]}\t{df1_cols[1]}\n"
        f"{dash}\n"
        f"{df1.at(0, 'Test')}\t{df1.at(0, 'Test2')}\t\n"
        f"{df1.at(1, 'Test')}\t{df1.at(1, 'Test2')}\t\n"
    )

    assert df1_string == repr(df1)

def test_dataframe_head():
    """Verify head() function of DataFrame"""
    df1 = DataFrame('Test', [1, 2])
    df1.add_column('Test2', [3, 4])
    df1_cols = df1.get_column_names()
    dash = "-" * 8 * df1.cols

    df1_string = (
        f"DataFrame ({df1.rows} rows x {df1.cols} cols)\n"
        f"{df1_cols[0]}\t{df1_cols[1]}\n"
        f"{dash}\n"
        f"{df1.at(0, 'Test')}\t{df1.at(0, 'Test2')}\t\n"
        f"{df1.at(1, 'Test')}\t{df1.at(1, 'Test2')}\t\n"
    )

    assert df1_string == repr(df1.head())

def test_dataframe_filter():
    """Verify DataFrame filter"""
    df1 = DataFrame('Test', [1, 2])
    df1.add_column('Test2', [3, 4])
    df1.add_column('Test3', [5, 6])
    df1.add_column('Test4', [7, 8])

    df2 = df1.filter('Test2', lambda x: x < 4)
    df2_cols = df2.get_column_names()
    dash = "-" * 8 * df1.cols

    df2_string = (
        f"DataFrame ({df2.rows} rows x {df2.cols} cols)\n"
        f"{df2_cols[0]}\t{df2_cols[1]}\t{df2_cols[2]}\t{df2_cols[3]}\n"
        f"{dash}\n"
        f"{df2.at(0, 'Test')}\t{df2.at(0, 'Test2')}\t{df2.at(0, 'Test3')}\t{df2.at(0, 'Test4')}\t\n"
    )

    assert df2_string == repr(df2)
    assert df2.get_column_names() == ['Test', 'Test2', 'Test3', 'Test4']
    assert df2.cols == 4
    assert df2.rows == 1

    with pytest.raises(ValueError):
        df1.filter('World', lambda x: x < 4)

    with pytest.raises(TypeError):
        df1.filter('Test2', 'hello')

def test_dataframe_remove_col():
    """Verify head() function of DataFrame"""
    df1 = DataFrame('Test', [1, 2])
    df1.add_column('Test2', [3, 4])

    df1.drop_column('Test')

    assert df1.get_column_names() == ['Test2']
    assert df1.cols == 1
    assert df1.rows == 2

    df2 = DataFrame('Test', [1, 2])
    df2.add_column('Test2', [3, 4])

    df2.drop_column('Test2')

    assert df2.get_column_names() == ['Test']
    assert df2.cols == 1
    assert df2.rows == 2

    with pytest.raises(ValueError):
        df1.drop_column('Hello')

def test_dataframe_encode_binary():
    """Test DataFrame Encode Binary function"""
    gender_list = []
    for i in range(10):
        prob = random.random()
        gender = 'Male' if prob > 0.5 else 'Female'
        gender_list.append(gender)
    nums = list(range(1, 11))

    df = DataFrame('ID', nums)
    df.add_column('Gender', gender_list)

    df.encode_binary('Gender', 'Male', 'Female')

    assert df.cols == 2
    assert df.rows == 10

    for i in range(10):
        assert df.at(i, 'Gender') == 0 or df.at(i, 'Gender') == 1

    gender_list = []
    for i in range(10):
        prob = random.random()
        gender = 'Male' if prob > 0.5 else 'Female'
        gender_list.append(gender)
    nums = list(range(1, 11))

    df2 = DataFrame('ID', nums)
    df2.add_column('Gender', gender_list)

    with pytest.raises(RuntimeError):
        df2.encode_binary('Gender', 'Male', 'Test')

    with pytest.raises(ValueError):
        df2.encode_binary('World', 'Male', 'Female')

def test_dataframe_to_matrix():
    """Test Dataframe to_matrix function."""
    gender_list = []
    for i in range(10):
        prob = random.random()
        gender = 'Male' if prob > 0.5 else 'Female'
        gender_list.append(gender)
    nums = list(range(1, 11))

    df = DataFrame('ID', nums)
    df.add_column('Gender', gender_list)

    df.encode_binary('Gender', 'Male', 'Female')

    m: Matrix = df.to_matrix(['ID', 'Gender'])

    assert m.cols == 2
    assert m.rows == 10

    for i in range(m.rows):
        for j in range(m.cols):
            assert m[i, j] == df.at(i, j)

    m2: Matrix = df.to_matrix(['Gender'])
    assert m2.cols == 1
    assert m2.rows == 10
    for i in range(m2.rows):
        assert m2[i, 0] == df.at(i, 1)

    with pytest.raises(TypeError):
        df.to_matrix('Gender')

    with pytest.raises(IndexError):
        df.to_matrix(['Hello'])