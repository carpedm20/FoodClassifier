package main

import (
    "image"
    "image/jpeg"
    "os"
    "fmt"
    "log"
    "bufio"
    "strings"
    "runtime"
    "strconv"
    "io/ioutil"

    "github.com/disintegration/gift"
)

func main() {
    runtime.GOMAXPROCS(runtime.NumCPU())

    //TRAIN_DATA_ROOT := "/home/carpedm20/data/food100/"
    TRAIN_DATA_ROOT := "/Users/carpedm20/data/food100/"

    //files := []string{"7.jpg", "30.jpg", "35.jpg"}
    files, _ := ioutil.ReadDir(TRAIN_DATA_ROOT)

    for _, parent := range files {
        if (parent.Name() != "Linux_doc" && parent.IsDir()) {
            current_dir := TRAIN_DATA_ROOT + "/" + parent.Name()
            crop_info, err := os.Open(current_dir + "/bb_info.txt")

            scanner := bufio.NewScanner(crop_info)
            crop_map := make(map[string][]string)

            for scanner.Scan() {
                crops := strings.Fields(scanner.Text())
                crop_map[crops[0]] = crops[1:]
            }

            fmt.Println(crop_map)

            p, err := ioutil.ReadDir(current_dir)

            if err != nil {
                panic(err)
            }

            for _, child := range p {
                if (child.Name() != "bb_info.txt") {
                    idx := strings.Index(child.Name(), ".jpg")
                    if idx == -1 {
                        panic("no *.jpg in " + child.Name())
                    }

                    c := crop_map[child.Name()[:idx]]
                    //fmt.Println(strconv.Atoi(c[0]))
                    fmt.Println(child.Name() + " : " + strings.Join(c, ", "))

                    f, err := os.Open(current_dir + child.Name())

                    if err != nil {
                        log.Fatal(err)
                    }

                    img, err := jpeg.Decode(f)

                    x0, _ := strconv.Atoi(c[0])
                    y0, _ := strconv.Atoi(c[1])
                    x1, _ := strconv.Atoi(c[2])
                    y1, _ := strconv.Atoi(c[3])

                    g := gift.New(
                        gift.Crop(image.Rect(x0, y0, x1, y1)),
                        //gift.Crop(image.Rect(0, 0, 10, 10)),
                    )

                    dst := image.NewRGBA(g.Bounds(img.Bounds()))
                    g.Draw(dst, img)

                    out, err := os.Create("crop_" + child.Name())
                    if err != nil {
                        log.Fatal(err)
                    }
                    defer out.Close()

                    jpeg.Encode(out, dst, nil)
                }
            }
        }
    }
}
