package main

import (
    "image"
    "image/jpeg"
    "os"
    "log"
    "runtime"
    "io/ioutil"

    "github.com/disintegration/gift"
)

func main() {
    runtime.GOMAXPROCS(runtime.NumCPU())

    TRAIN_DATA_ROOT := "/home/carpedm20/data/food100/"
    PATH := "/Users/carpedm20/data/1/"

    //files := []string{"7.jpg", "30.jpg", "35.jpg"}
    files, _ := ioutil.ReadDir(TRAIN_DATA_ROOT)

    for _, parent := range files {
        if (parent.Name() != "Linux_doc" && parent.IsDir()) {
            p, _ := ioutil.ReadDir(buffer.String())

            crop_info, err := os.Open(parent + "bb_info.txt")

            if err != nil {
                panic(err)
            }

            for _, child := range p {
                if (child.Name() != "bb_info.txt") {
                    for _, file := range files {
                        f, err := os.Open(child)

                        if err != nil {
                            log.Fatal(err)
                        }

                        img, err := jpeg.Decode(f)

                        g := gift.New(
                            gift.Crop(image.Rect(100, 100, 200, 200)),
                        )

                        dst := image.NewRGBA(g.Bounds(img.Bounds()))
                        g.Draw(dst, img)

                        out, err := os.Create("crop_" + file)
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
}
