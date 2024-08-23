import math

lattice_segments = None


class SegmentsLattice:

    def __init__(self, regions, align_point_index):
        self.align_point_index = align_point_index
        root_size = len(regions)
        self.structure = []
        self.structure.append(regions)
        for i in range(0, len(self.structure[0])):
            self.structure[0][i] = (self.structure[0][i][0] + align_point_index, self.structure[0][i][1] + align_point_index)
        for line in range(0, root_size - 1):
            tmp = []
            for column in range(0, root_size-line-1):
                new = (self.structure[line][column][0], self.structure[line][column+1][1])
                tmp.append(new)
            self.structure.append(tmp)

    def get_max_rank(self):
        return len(self.structure) - 1

    def get_bounds(self, rank, index):
        return self.structure[rank][index][0]-self.align_point_index, self.structure[rank][index][1]-self.align_point_index

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        max_e = -1
        for line in range(0, len(self.structure)):
            for column in range(0, len(self.structure[line])):
                tmp = '[' + str(self.structure[line][column][0]-self.align_point_index) + ':' + str(self.structure[line][column][1]-self.align_point_index) + ']'
                max_e = max(len(tmp), max_e)
        ret = ''
        first_line = True
        max_l = -1
        for line in range(0, len(self.structure)):
            tmp = ''
            for column in range(0, len(self.structure[line])):
                tmp += ('[' + str(self.structure[line][column][0]-self.align_point_index) + ':' + str(self.structure[line][column][1]-self.align_point_index) + ']').center(max_e)
            tmp = tmp.strip()
            if first_line:
                first_line = False
                max_l = len(tmp)
            tmp = tmp.center(max_l)
            ret = tmp + '\n' + ret
        return ret


def generate_segment_lattice(number_of_regions, sequence_size, alignement_point):
    if number_of_regions % 2 == 0:
        number_of_regions += 1
    left_limit = 0 - alignement_point
    right_limit = sequence_size - alignement_point - 1
    left_part_size = alignement_point
    right_part_size = sequence_size - alignement_point

    central_region = (0, 0)

    left_regions = []
    right_regions = []
    done = False
    k_init = 0
    n = 3
    while not done:
        left_regions = []
        right_regions = []
        remaining_left_part_size = left_part_size
        remaining_right_part_size = right_part_size
        left_regions.append(central_region)
        right_regions.append(central_region)
        k_init += 1
        k = k_init
        while remaining_left_part_size > 0 or remaining_right_part_size > 0:
            k += 1
            if remaining_left_part_size:
                last_left_region = left_regions[-1]
                new_left_region = (last_left_region[0]-k**n, last_left_region[0])
                left_regions.append(new_left_region)
                remaining_left_part_size -= (new_left_region[1] - new_left_region[0])
                if remaining_left_part_size <= 0:
                    left_regions[-1] = (left_limit, left_regions[-1][1])
            if remaining_right_part_size:
                last_right_region = right_regions[-1]
                new_right_region = (last_right_region[1], last_right_region[1]+k**n)
                right_regions.append(new_right_region)
                remaining_right_part_size -= (new_right_region[1] - new_right_region[0])
                if remaining_right_part_size <= 0:
                    right_regions[-1] = (right_regions[-1][0], right_limit)
        done = (len(left_regions) + len(right_regions) - 1) <= number_of_regions

    left_regions = left_regions[1:]
    right_regions = right_regions[1:]
    
    if right_regions[-2][1] == sequence_size - alignement_point - 1:
        right_regions = right_regions[:-1]
    
    for i in range(0, len(left_regions)):
        left_regions[i] = (left_regions[i][0], left_regions[i][1]-1)
    for i in range(0, len(right_regions)):
        right_regions[i] = (right_regions[i][0]+1, right_regions[i][1])
    regions = left_regions + [central_region] + right_regions

    removable_regions = []
    for r in regions:
        if r[0] > r[1]:
            removable_regions.append(r)
    for r in removable_regions:
        regions.remove(r)

    regions = sorted(regions, key=lambda tup: (tup[0], tup[1]))
    lattice = SegmentsLattice(regions, alignement_point)
    return lattice


def generate_segment_lattice_uniform(number_of_regions, sequence_size, alignement_point):
    if number_of_regions % 2 == 0:
        number_of_regions += 1
    left_limit = 0 - alignement_point
    right_limit = sequence_size - alignement_point - 1
    left_part_size = alignement_point
    right_part_size = sequence_size - alignement_point

    central_region = (0, 0)

    left_regions = []
    right_regions = []
    done = False
    region_size = math.floor(sequence_size / (number_of_regions - 1))
    while not done:
        left_regions = []
        right_regions = []
        remaining_left_part_size = left_part_size
        remaining_right_part_size = right_part_size
        left_regions.append(central_region)
        right_regions.append(central_region)
        while remaining_left_part_size > 0 or remaining_right_part_size > 0:
            if remaining_left_part_size:
                last_left_region = left_regions[-1]
                new_left_region = (last_left_region[0]-region_size, last_left_region[0])
                left_regions.append(new_left_region)
                remaining_left_part_size -= (new_left_region[1] - new_left_region[0])
                if remaining_left_part_size <= 0:
                    left_regions[-1] = (left_limit, left_regions[-1][1])
            if remaining_right_part_size:
                last_right_region = right_regions[-1]
                new_right_region = (last_right_region[1], last_right_region[1]+region_size)
                right_regions.append(new_right_region)
                remaining_right_part_size -= (new_right_region[1] - new_right_region[0])
                if remaining_right_part_size <= 0:
                    right_regions[-1] = (right_regions[-1][0], right_limit)
        done = (len(left_regions) + len(right_regions) - 1) >= number_of_regions

    left_regions = left_regions[1:]
    right_regions = right_regions[1:]

    if right_regions[-2][1] == sequence_size - alignement_point - 1:
        right_regions = right_regions[:-1]

    for i in range(0, len(left_regions)):
        left_regions[i] = (left_regions[i][0], left_regions[i][1]-1)
    for i in range(0, len(right_regions)):
        right_regions[i] = (right_regions[i][0]+1, right_regions[i][1])
    regions = left_regions + [central_region] + right_regions

    removable_regions = []
    for r in regions:
        if r[0] > r[1]:
            removable_regions.append(r)
    for r in removable_regions:
        regions.remove(r)

    regions = sorted(regions, key=lambda tup: (tup[0], tup[1]))
    lattice = SegmentsLattice(regions, alignement_point)
    return lattice


def generate_segment_lattice_given_bins(bins, alignement_point):
    regions = []
    bins = [int(_) for _ in bins]
    bins = sorted(bins)
    for i in range(len(bins)-1):
        regions.append((bins[i], bins[i+1]-1))
    lattice = SegmentsLattice(regions, alignement_point)
    return lattice


if __name__ == '__main__':
    import sys
    print(generate_segment_lattice(number_of_regions=int(sys.argv[1]), sequence_size=10000, alignement_point=5000))
