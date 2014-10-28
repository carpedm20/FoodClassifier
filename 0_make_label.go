package main

import (
  "fmt"
  "bufio"
  "io/ioutil"
  "os"
  "bytes"
  "math/rand"
)


func Shuffle(a []os.FileInfo) {
  for i := range a {
    j := rand.Intn(i + 1)
    a[i], a[j] = a[j], a[i]
  }
}

func main() {
  TRAIN_DATA_ROOT := "/home/carpedm20/data/food100/"
  //VAL_DATA_ROOT := "/home/carpedm20/data/food100/"

  TRAIN_FILE := "../../data/food100/train.txt"
  VAL_FILE := "../../data/food100/val.txt"

  train_f, _ := os.Create(TRAIN_FILE)
  train_w := bufio.NewWriter(train_f)

  val_f, _ := os.Create(VAL_FILE)
  val_w := bufio.NewWriter(val_f)

  files, _ := ioutil.ReadDir(TRAIN_DATA_ROOT)
  for _, parent := range files {
    if (parent.Name() != "Linux_doc" && parent.IsDir()) {
      var buffer bytes.Buffer

      buffer.WriteString(TRAIN_DATA_ROOT)
      buffer.WriteString("/")
      buffer.WriteString(parent.Name())

      p, _ := ioutil.ReadDir(buffer.String())
      Shuffle(p)

      slicePoint := int(float64(len(p))*0.8)

      for _, child := range p[:slicePoint] {
        if (child.Name() != "bb_info.txt") {
          fmt.Printf("%s\n", child.Name())

          str := fmt.Sprintf("%s/%s %s\n", parent.Name(), child.Name(), parent.Name())
          train_w.WriteString(str)
        }
      }

      for _, child := range p[slicePoint:] {
        if (child.Name() != "bb_info.txt") {
          fmt.Printf("%s\n", child.Name())

          str := fmt.Sprintf("%s/%s %s\n", parent.Name(), child.Name(), parent.Name())
          val_w.WriteString(str)
        }
      }
    }
  }

  train_w.Flush()
  val_w.Flush()
}
