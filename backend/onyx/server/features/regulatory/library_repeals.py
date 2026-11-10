"""Official repeal evidence verified against exact retained document revisions."""

from uuid import UUID

from onyx.server.features.regulatory.library_models import LibraryRepeal

_LAW_5911 = (
    "https://cdn.tbmm.gov.tr/KKBSPublicFile/D23/Y2/T1/KanunMetni/"
    "ae4c9774-655b-4fd2-942e-6f8f7d28969a.html"
)
_LAW_5728 = (
    "https://www5.tbmm.gov.tr/tutanaklar/KANUNLAR_KARARLAR/"
    "kanuntbmmc092/kanuntbmmc092/kanuntbmmc09205728.pdf"
)
_VERIFIED_REPEALS = {
    (
        UUID("12328001-2f21-407e-b230-7d50a8dbf28f"),
        "1db892a36effeeb8bf6e56871448b6148a935354fd788f7a8b2e555fe4deff64",
    ): (
        *(
            LibraryRepeal(
                heading=f"MADDE {number}-",
                source_label="5911 m. 68/A",
                source_url=_LAW_5911,
            )
            for number in (42, 43, 44, 240)
        ),
        LibraryRepeal(
            heading="Madde 222. -",
            source_label="666 KHK, ek m. 12/1-f",
            source_url="https://resmigazete.gov.tr/eskiler/2011/11/20111102M1-6.htm",
        ),
        LibraryRepeal(
            heading="ı)",
            source_label="7190 m. 15 · 4458 m. 241/4-ı",
            source_url=(
                "https://cdn.tbmm.gov.tr/KKBSPublicFile/D27/Y3/T2/KanunMetni/"
                "f448af71-9223-4afb-be6b-6d5c0557f0f8.html"
            ),
            kind="clause",
        ),
    ),
    (
        UUID("93135f0c-7c7e-4481-a345-c66d84fde62a"),
        "3a94984c86912f3afc1c3e397525e4cc8e8ce8e22e65157ef09df38650d32608",
    ): tuple(
        LibraryRepeal(
            heading=f"Madde {number} -",
            source_label="5728 m. 578",
            source_url=_LAW_5728,
        )
        for number in (3, 4)
    ),
    (
        UUID("d0c8d7b3-072c-4c2f-b550-17fac959cf3c"),
        "0c76a1741626b9924b4b63ee150cb21860c65456ee3caf90107e6ce8382e3e4b",
    ): (
        LibraryRepeal(
            heading="h) ,",
            source_label="RG 28.09.2014/29133 m. 3",
            source_url="https://www.resmigazete.gov.tr/eskiler/2014/09/20140928-1.htm",
            kind="clause",
        ),
    ),
    (
        UUID("21b49197-b281-484c-a464-b2b0151d0d27"),
        "e6e167ecc4567f2472d81f7c03d63ae3d6eaca5025d1688ba0e633dbb2d61653",
    ): (
        LibraryRepeal(
            heading="f) .......................",
            source_label="4369 m. 82/5 · 3065 m. 10/f",
            source_url="https://gib.gov.tr/mevzuat/kanun/436/madde/7905",
            kind="clause",
        ),
        LibraryRepeal(
            heading="h) ........................",
            source_label="3316 m. 2 · 3065 m. 10/h",
            source_url="https://gib.gov.tr/mevzuat/kanun/436/madde/7905",
            kind="clause",
        ),
    ),
    (
        UUID("42ce7902-40d1-4f45-841a-d1578a033829"),
        "87e72f7f810a3300d098541153dfa4a47b84ad61be2e7a5bda7bd7dd1784f846",
    ): (
        LibraryRepeal(
            heading="h)",
            source_label="RG 08.04.2017/30032 m. 7 · Limanlar Yönetmeliği m. 20/1-h",
            source_url="https://www.resmigazete.gov.tr/eskiler/2017/04/20170408-4.htm",
            kind="clause",
        ),
    ),
}


def verified_repeals(file_id: UUID, content_sha256: str) -> list[LibraryRepeal]:
    return list(_VERIFIED_REPEALS.get((file_id, content_sha256), ()))
