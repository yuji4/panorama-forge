import cv2
import numpy as np
import os
import sys
import argparse
import glob


# 1. 특징점 검출 및 매칭

def detect_and_match(img1, img2, ratio_thresh=0.75):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, None

    # FLANN기반 매처 설정
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test로 오매칭 제거
    good = []
    for m_n in raw_matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good.append(m)

    if len(good) < 4:
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


# 2. 호모그래피 추정 (RANSAC)

def estimate_homography(pts1, pts2, ransac_thresh=4.0):
    if pts1 is None or len(pts1) < 4:
        return None
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None:
        return None
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"  [Homography] Inliers: {inliers}/{len(pts1)}")
    return H


# 3. 멀테밴드 블렌딩

def build_gaussian_pyramid(img, levels):
    gp = [img.astype(np.float32)]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        gp.append(img.astype(np.float32))
    return gp


def build_laplacian_pyramid(img, levels):
    gp = build_gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels - 1):
        up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lap = gp[i].astype(np.float32) - up.astype(np.float32)
        lp.append(lap)
    lp.append(gp[-1].astype(np.float32))
    return lp


def multiband_blend(img1, img2, mask1, mask2, levels=5):

    h = max(img1.shape[0], img2.shape[0])
    w = max(img1.shape[1], img2.shape[1])

    def pad(img, th, tw):
        out = np.zeros((th, tw, img.shape[2]), dtype=np.float32)
        out[:img.shape[0], :img.shape[1]] = img.astype(np.float32)
        return out

    def pad_mask(m, th, tw):
        out = np.zeros((th, tw), dtype=np.float32)
        out[:m.shape[0], :m.shape[1]] = m.astype(np.float32)
        return out

    i1 = pad(img1, h, w)
    i2 = pad(img2, h, w)
    m1 = pad_mask(mask1, h, w)
    m2 = pad_mask(mask2, h, w)

    # 가중치 정규화
    total = m1 + m2 + 1e-8
    w1 = m1 / total
    w2 = m2 / total

    lp1 = build_laplacian_pyramid(i1, levels)
    lp2 = build_laplacian_pyramid(i2, levels)
    gp_w1 = build_gaussian_pyramid(w1, levels)
    gp_w2 = build_gaussian_pyramid(w2, levels)

    blended_lp = []
    for la, lb, wa, wb in zip(lp1, lp2, gp_w1, gp_w2):
        if la.ndim == 3:
            wa = wa[..., np.newaxis]
            wb = wb[..., np.newaxis]
        blended_lp.append(la * wa + lb * wb)

    # 블렌딩된 라플라시안 피라미드로 이미지 복원
    result = blended_lp[-1]
    for lap in reversed(blended_lp[:-1]):
        result = cv2.pyrUp(result, dstsize=(lap.shape[1], lap.shape[0]))
        result = result + lap

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result[:h, :w]


# 4. 이미지 워핑 및 스티칭

def warp_and_stitch(base, new_img, H):

    h1, w1 = base.shape[:2]
    h2, w2 = new_img.shape[:2]

    # 워핑 후 new_img의 네 꼭짓점 좌표 계산
    corners_new = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_new, H)

    # base 이미지 꼭짓점과 합쳐서 출력 캔버스 크기 결정
    corners_base = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([corners_base, warped_corners], axis=0)

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    tx, ty = -x_min, -y_min
    translation = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

    out_w = x_max - x_min
    out_h = y_max - y_min

    # new_img 워핑
    warped_new = cv2.warpPerspective(new_img, translation @ H, (out_w, out_h))
    warped_new_mask = cv2.warpPerspective(
        np.ones((h2, w2), dtype=np.float32), translation @ H, (out_w, out_h)
    )

    # base 이미지를 캔버스에 배치
    base_canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    base_canvas[ty:ty + h1, tx:tx + w1] = base
    base_mask = np.zeros((out_h, out_w), dtype=np.float32)
    base_mask[ty:ty + h1, tx:tx + w1] = 1.0

    # 멀티밴드 블렌딩으로 합성
    result = multiband_blend(base_canvas, warped_new, base_mask, warped_new_mask, levels=5)
    return result


# 5. 이미지 순서 결정

def order_images(images):
    
    n = len(images)
    if n <= 1:
        return list(range(n)), [None]

    # 모든 이미지 쌍의 매칭 수 행렬 계산
    match_count = np.zeros((n, n), dtype=int)
    print("Computing pairwise matches for ordering...")
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = detect_and_match(
                cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(images[j], cv2.COLOR_BGR2GRAY),
            )
            cnt = len(p1) if p1 is not None else 0
            match_count[i][j] = cnt
            match_count[j][i] = cnt

    # 총 매칭 수가 가장 많은 이미지를 시작점으로 선택 (중앙에 가까운 이미지)
    start = int(np.argmax(match_count.sum(axis=1)))
    ordered = [start]
    remaining = list(range(n))
    remaining.remove(start)

    while remaining:
        last = ordered[-1]
        best = max(remaining, key=lambda j: match_count[last][j])
        ordered.append(best)
        remaining.remove(best)

    return ordered


# 6. 메인 스티칭 파이프라인

def stitch_images(image_paths, output_path="panorama.jpg"):
    print(f"\n{'='*50}")
    print(f"PanoramaForge - Loading {len(image_paths)} images")
    print(f"{'='*50}")

    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  [WARN] Cannot read: {p}")
            continue
        # 처리 속도를 위해 큰 이미지는 리사이즈
        max_dim = 1200
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        images.append(img)
        print(f"  Loaded: {p} ({img.shape[1]}x{img.shape[0]})")

    if len(images) < 2:
        print("ERROR: Need at least 2 valid images.")
        return None

    # 스티칭 순서 결정
    order = order_images(images)
    print(f"\nImage order: {order}")

    # 매칭용 그레이스케일 변환
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # 첫 번째 이미지를 기준 파노라마로 설정
    panorama = images[order[0]]
    print(f"\nStarting with image index {order[0]}")

    for i in range(1, len(order)):
        idx = order[i]
        prev_idx = order[i - 1]
        print(f"\n[Step {i}] Stitching image {idx} onto panorama...")

        # 현재 파노라마와 새 이미지 간의 특징점 매칭
        gray_pano = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        gray_new = grays[idx]

        pts_pano, pts_new = detect_and_match(gray_pano, gray_new)

        if pts_pano is None:
            print(f"  [WARN] Not enough matches for image {idx}, skipping.")
            continue

        print(f"  Matches found: {len(pts_pano)}")

        # 호모그래피 추정: new_img -> 파노라마 좌표계
        H = estimate_homography(pts_new, pts_pano)
        if H is None:
            print(f"  [WARN] Homography failed for image {idx}, skipping.")
            continue

        panorama = warp_and_stitch(panorama, images[idx], H)
        print(f"  Panorama size: {panorama.shape[1]}x{panorama.shape[0]}")

    # 검은 테두리 제거
    panorama = crop_black_border(panorama)

    cv2.imwrite(output_path, panorama)
    print(f"\n{'='*50}")
    print(f"Panorama saved: {output_path}  ({panorama.shape[1]}x{panorama.shape[0]})")
    print(f"{'='*50}\n")
    return panorama


def crop_black_border(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img[y:y + h, x:x + w]

# 7. 데모: 테스트용 이미지 자동 생성

def generate_test_images(output_dir="test_images"):
    os.makedirs(output_dir, exist_ok=True)
    # 넓은 합성 장면 생성
    wide = np.zeros((400, 1200, 3), dtype=np.uint8)
    # 배경 그라디언트
    for x in range(1200):
        wide[:, x] = [int(x * 0.2), int(x * 0.05), int(200 - x * 0.15)]
    # 랜드마크 역할의 도형 추가
    cv2.rectangle(wide, (100, 100), (200, 300), (255, 100, 50), -1)
    cv2.circle(wide, (400, 200), 80, (50, 255, 100), -1)
    cv2.rectangle(wide, (600, 80), (750, 320), (100, 50, 255), -1)
    cv2.circle(wide, (900, 200), 70, (255, 255, 50), -1)
    cv2.rectangle(wide, (1050, 120), (1150, 280), (50, 200, 255), -1)
    # 노이즈 추가
    noise = np.random.randint(0, 30, wide.shape, dtype=np.uint8)
    wide = cv2.add(wide, noise)

    paths = []
    # 40% 오버랩으로 3장의 스트립 생성
    overlap = 160
    strip_w = 480
    for i in range(3):
        start = i * (strip_w - overlap)
        end = start + strip_w
        strip = wide[:, start:min(end, 1200)]
        path = os.path.join(output_dir, f"test_{i+1}.jpg")
        cv2.imwrite(path, strip)
        paths.append(path)
        print(f"  Generated: {path}")
    return paths


# 8. CLI 진입점

def main():
    parser = argparse.ArgumentParser(
        description="PanoramaForge - Automatic Image Stitching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_stitching.py img1.jpg img2.jpg img3.jpg
  python image_stitching.py --dir ./my_photos --output panorama.jpg
  python image_stitching.py --demo
        """
    )
    parser.add_argument("images", nargs="*", help="Input image files")
    parser.add_argument("--dir", type=str, help="Directory containing images")
    parser.add_argument("--output", type=str, default="panorama.jpg", help="Output file path")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic images")

    args = parser.parse_args()

    if args.demo:
        print("Generating demo test images...")
        paths = generate_test_images()
    elif args.dir:
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        paths = []
        for ext in exts:
            paths.extend(sorted(glob.glob(os.path.join(args.dir, ext))))
        if not paths:
            print(f"No images found in: {args.dir}")
            sys.exit(1)
    elif args.images:
        paths = args.images
    else:
        print("No input specified. Run with --demo or provide image paths.")
        parser.print_help()
        sys.exit(1)

    if len(paths) < 2:
        print("ERROR: Need at least 2 images to stitch.")
        sys.exit(1)

    result = stitch_images(paths, args.output)

    if result is not None:
        
        cv2.imshow("PanoramaForge - Result", result)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()