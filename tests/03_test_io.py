from daedalus import read_csv, DataFrame

def test_read_csv():
    """Test read_csv function"""
    df: DataFrame = read_csv('tests/test.csv')

    assert df.get_column_names() == ['Test1', 'Test2', 'Test3']
    assert df.cols == 3
    assert df.rows == 2

    count = 1
    for i in range(df.rows):
        for j in range(df.cols):
            assert df.at(i, j) == count
            count += 1