from jinja2 import Environment, FileSystemLoader


def html_gene(
        model, train_img, validation_img, tv_rate, paper_url,
        paper_name, code_url, mIoU, mDSC, loss, train_time, images
):
    """
    Generate a html output
    :param model:
    :param train_img:
    :param validation_img:
    :param tv_rate:
    :param paper_url:
    :param paper_name:
    :param code_url:
    :param mIoU:
    :param mDSC:
    :param loss:
    :param train_time:
    :param images:
    :return:
    """
    # Set the environment
    env = Environment(loader=FileSystemLoader("./results/"))
    # Get template
    template = env.get_template("train_output_template.html")
    # Render the template
    with open("./results/train_output.html", "w+") as out_html:
        html_content = template.render(
            model=model,
            train_img=train_img,
            validation_img=validation_img,
            tv_rate=tv_rate,
            paper_url=paper_url,
            paper_name=paper_name,
            code_url=code_url,
            mIoU=mIoU,
            mDSC=mDSC,
            loss=loss,
            train_time=train_time
        )
        out_html.write(html_content)

    template = env.get_template("images_output_template.html")
    with open("results/images_output.html", "w+") as out_html:
        html_content = template.render(
            images=images
        )
        out_html.write(html_content)


def image_gene(imageID, image, label, IoU, DSC):
    """
    Generate a image row
    :param imageID:
    :param image:
    :param label:
    :param IoU:
    :param DSC:
    :return:
    """
    return {
        "imageID": imageID,
        "image": image,
        "label": label,
        "IoU": IoU,
        "DSC": DSC
    }


def densASPP_test():
    images = []
    image = image_gene(
        "000001",
        "../resources/data/aachen_000000_000019_leftImg8bit.png",
        "../resources/target/aachen_000000_000019_gtFine_color.png",
        "70%",
        "80%"
    )
    images.append(image)
    html_gene(
        "DenseASPP",
        3000,
        2000,
        1.5,
        "https://github.com/Ian-Dx/SidewalkDetection/blob/"
        "master/references/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf",
        "DenseASPP for Semantic Segmentation in Street Scenes",
        "https://github.com/Ian-Dx/SidewalkDetection/tree/master/SidewalkDetection",
        "85%",
        "84%",
        100,
        12121,
        images
    )


if __name__ == '__main__':
    densASPP_test()
