package main

import (
    "image"
    "image/jpeg"
    "os"
    "fmt"
    "bufio"
    "strings"
    "runtime"
    "strconv"
    "io/ioutil"

    "github.com/disintegration/gift"
)

func crop(img_name string, crop_map map[string][]string, current_dir string) {
    idx := strings.Index(img_name, ".jpg")
    if idx == -1 {
        panic("no *.jpg in " + img_name)
    }

    c := crop_map[img_name[:idx]]
    fmt.Println(img_name + " : " + strings.Join(c, ", "))

    f_path := current_dir + "/" + img_name

    f, _ := os.Open(f_path)
    defer f.Close()

    img, err := jpeg.Decode(f)
    if err != nil {
        return
    }

    /**********
     *  Crop
     **********/

    x0, _ := strconv.Atoi(c[0])
    y0, _ := strconv.Atoi(c[1])
    x1, _ := strconv.Atoi(c[2])
    y1, _ := strconv.Atoi(c[3])

    g := gift.New(
        gift.Crop(image.Rect(x0, y0, x1, y1)),
    )

    bounds := g.Bounds(img.Bounds())
    crop := image.NewRGBA(bounds)
    g.Draw(crop, img)

    crop_f_path := current_dir + "/crop_" + img_name
    out, _ := os.Create(crop_f_path)
    defer out.Close()

    jpeg.Encode(out, crop, nil)

    /**********
     *  Flip
     **********/

    g = gift.New(
        gift.FlipHorizontal(),
    )

    dst := image.NewRGBA(g.Bounds(crop.Bounds()))
    g.Draw(dst, crop)

    flip_f_path := current_dir + "/flip_" + img_name
    out, _ = os.Create(flip_f_path)
    defer out.Close()

    jpeg.Encode(out, dst, nil)

    /***********
     *  Rotate
     ***********/

    g = gift.New(
        gift.Rotate90(),
    )

    dst = image.NewRGBA(g.Bounds(crop.Bounds()))
    g.Draw(dst, crop)

    rotate_f_path := current_dir + "/rotate90_" + img_name
    out, _ = os.Create(rotate_f_path)
    defer out.Close()

    jpeg.Encode(out, dst, nil)

    g = gift.New(
        gift.Rotate180(),
    )

    dst = image.NewRGBA(g.Bounds(crop.Bounds()))
    g.Draw(dst, crop)

    rotate_f_path = current_dir + "/rotate180_" + img_name
    out, _ = os.Create(rotate_f_path)
    defer out.Close()

    jpeg.Encode(out, dst, nil)

    g = gift.New(
        gift.Rotate270(),
    )

    dst = image.NewRGBA(g.Bounds(crop.Bounds()))
    g.Draw(dst, crop)

    rotate_f_path = current_dir + "/rotate270_" + img_name
    out, _ = os.Create(rotate_f_path)
    defer out.Close()

    jpeg.Encode(out, dst, nil)
}

func stringInSlice(a string, list []string) bool {
    for _, b := range list {
        if strings.Index(a,b) != -1 {
            return true
        }
    }
    return false
}

func main() {
    runtime.GOMAXPROCS(runtime.NumCPU())

    TRAIN_DATA_ROOT := "/home/carpedm20/data/food100"
    //TRAIN_DATA_ROOT := "/Users/carpedm20/data/food100"

    //files := []string{"7.jpg", "30.jpg", "35.jpg"}
    files, _ := ioutil.ReadDir(TRAIN_DATA_ROOT)

    ban_list := []string {"crop", "rotate", "flip"}

    for _, parent := range files {
        if (parent.Name() != "Linux_doc" && parent.IsDir()) {
            current_dir := TRAIN_DATA_ROOT + "/" + parent.Name()
            //current_dir := TRAIN_DATA_ROOT + "/21"
            fmt.Println("[*] Start : " + current_dir)
            crop_info, err := os.Open(current_dir + "/bb_info.txt")

            scanner := bufio.NewScanner(crop_info)
            crop_map := make(map[string][]string)

            for scanner.Scan() {
                crops := strings.Fields(scanner.Text())
                crop_map[crops[0]] = crops[1:]
            }

            crop_info.Close()

            p, err := ioutil.ReadDir(current_dir)

            if err != nil {
                panic(err)
            }

            for _, child := range p {
                f_name := child.Name()
                if (f_name != "bb_info.txt" && !stringInSlice(f_name, ban_list)) {
                    crop(child.Name(), crop_map, current_dir)
                }
            }
        }
    }

    fmt.Println(" === Finished === ")
}
