const fs = require('fs')
const sqlite3 = require('sqlite3')

const db = new sqlite3.Database('database.sqlite')

db.serialize(() => {
  db.each("SELECT Id, RawText FROM emails", function(err, row) {
    const filename = `data/${ row.Id }.txt`
//    console.log(row.id + ": " + row.RawText);
    console.log(filename)
    fs.writeFileSync(filename, row.RawText)

  });
});

db.close();

