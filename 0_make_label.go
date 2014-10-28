package main

import (
	"fmt"
  "bufio"
	"io/ioutil"
  "os"
  "bytes"
)

func main() {
  TRAIN_DATA_ROOT := "/home/carpedm20/data/food100/"
  //VAL_DATA_ROOT := "/home/carpedm20/data/food100/"

  TEST_FILE := "../../data/food100/test.txt"

  f, _ := os.Create(TEST_FILE)
  w := bufio.NewWriter(f)

	files, _ := ioutil.ReadDir(TRAIN_DATA_ROOT)
	for _, parent := range files {
    if (parent.Name() != "Linux_doc" && parent.IsDir()) {
      var buffer bytes.Buffer
      buffer.WriteString(TRAIN_DATA_ROOT)
      buffer.WriteString("/")
      buffer.WriteString(parent.Name())

      p, _ := ioutil.ReadDir(buffer.String())

      for _, child := range p {
        if (child.Name() != "bb_info.txt") {
          fmt.Printf("%s\n", child.Name())

          str := fmt.Sprintf("%s/%s %s\n", parent.Name(), child.Name(), parent.Name())
          w.WriteString(str)
        }
      }
    }
	}

  w.Flush()
}
