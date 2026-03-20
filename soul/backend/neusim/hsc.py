import numpy as np
import matplotlib.pyplot as plt


class HSCurve(object):

    def __init__(self,
                 width,
                 height):
        self.x = width
        self.y = height
        self.x_old = 0
        self.y_old = 0
        self.seg_list = np.zeros(shape=(width*height, 2), dtype=np.int32)
        self.ordered_points = None
        self.count = 0

    def compute_sfc(self):
        self.space_fill(1, 1, self.x, self.y)
        # self.ordered_points = [[x1-1, y1-1] for x1, y1, _, _, _ in self.seg_list]
        # x, y = self.seg_list[-1][2:4]
        # self.ordered_points.append([x-1, y-1])
        # self.ordered_points = np.array(self.ordered_points)
        ii = 0 if self.x==1 and self.y==1 else 1
        self.ordered_points = self.seg_list - ii

    def render(self, x0, y0, dir):
        self.x_new = x0
        self.y_new = y0

        if self.x_old > 0 and self.x_new > 0:
            # if dir == 'm':
                # self.seg_list.append([self.x_old, self.y_old, self.x_new, self.y_new, 'r'])
            self.seg_list[self.count, :] = [int(self.x_old), int(self.y_old)]
            self.count += 1
            if self.count == self.x*self.y-1:
                self.seg_list[self.count] = [int(self.x_new), int(self.y_new)]
            # else:
            #     self.seg_list.append([self.x_old, self.y_old, self.x_new, self.y_new, 'b'])

        self.x_old = self.x_new
        self.y_old = self.y_new

        return

    def space_fill(self, ll, tt, ww, hh):  # //left, top, width, height
        if (hh > ww):  # //go top->down
            if ((hh % 2 == 1) and (ww % 2 == 0)):
                self.go(ll, tt, ww, 0, 0, hh, "m")  # //go diagonal
            else:
                self.go(ll, tt, ww, 0, 0, hh, "r")  # //go top->down

        else:  # //go left->right
            if ((ww % 2 == 1) and (hh % 2 == 0)):
                self.go(ll, tt, ww, 0, 0, hh, "m")  # //go diagonal
            else:
                self.go(ll, tt, ww, 0, 0, hh, "l")  # //go left->right

    def go(self, x0, y0, dxl, dyl, dxr, dyr, dir):
        # x0, y0: start corner looking to the center of the rectangle
        # dxl, dyl: vector from the start corner to the left corner of the rectangle
        # dxr, dyr: vector from the start corner to the right corner of the rectangle
        # dir: direction to go - "l"=left, "m"=middle, "r"=right
        # msg("go: "+x0+", "+y0+", "+dxl+", "+dyl+", "+dxr+", "+dyr+", "+dir)
        # self.render if 2x3 or smaller
        if (abs((dxl + dyl) * (dxr + dyr)) <= 6):
            if (abs(dxl + dyl) == 1):
                ddx = dxr / abs(dxr + dyr)
                ddy = dyr / abs(dxr + dyr)
                for ii in range(0, abs(dxr + dyr)):
                    # for (ii=0 ii<abs(dxr+dyr) ii++)
                    self.render(x0 + ii * ddx + (dxl + ddx - 1) / 2, y0 + ii * ddy + (dyl + ddy - 1) / 2, dir)
                return

            if (abs(dxr + dyr) == 1):
                ddx = dxl / abs(dxl + dyl)
                ddy = dyl / abs(dxl + dyl)
                for ii in range(0, abs(dxl + dyl)):
                    # for (ii=0 ii<abs(dxl+dyl) ii++)
                    self.render(x0 + ii * ddx + (dxr + ddx - 1) / 2, y0 + ii * ddy + (dyr + ddy - 1) / 2, dir)
                return

            if (dir == "l"):
                ddx = dxr / abs(dxr + dyr)
                ddy = dyr / abs(dxr + dyr)
                for ii in range(0, abs(dxr + dyr)):
                    # for (ii=0 ii<abs(dxr+dyr) ii++)
                    self.render(x0 + ii * ddx + (dxl / 2 + ddx - 1) / 2, y0 + ii * ddy + (dyl / 2 + ddy - 1) / 2, dir)

                for ii in range(abs(dxr + dyr) - 1, -1, -1):
                    # for (ii=abs(dxr+dyr)-1 ii>=0 ii--)
                    self.render(x0 + ii * ddx + (dxl + dxl / 2 + ddx - 1) / 2,
                                y0 + ii * ddy + (dyl + dyl / 2 + ddy - 1) / 2, dir)
                return

            if (dir == "r"):
                ddx = dxl / abs(dxl + dyl)
                ddy = dyl / abs(dxl + dyl)
                for ii in range(0, abs(dxl + dyl)):
                    # for (ii=0 ii<abs(dxl+dyl) ii++)
                    self.render(x0 + ii * ddx + (dxr / 2 + ddx - 1) / 2, y0 + ii * ddy + (dyr / 2 + ddy - 1) / 2, dir)
                for ii in range(abs(dxl + dyl) - 1, -1, -1):
                    # for (ii=abs(dxl+dyl)-1 ii>=0 ii--)
                    self.render(x0 + ii * ddx + (dxr + dxr / 2 + ddx - 1) / 2,
                                y0 + ii * ddy + (dyr + dyr / 2 + ddy - 1) / 2, dir)
                return

            if (dir == "m"):
                if (abs(dxr + dyr) == 3):
                    ddx = dxr / abs(dxr + dyr)
                    ddy = dyr / abs(dxr + dyr)
                    self.render(x0 + (dxl / 2 + ddx - 1) / 2, y0 + (dyl / 2 + ddy - 1) / 2, dir)
                    self.render(x0 + (dxl + dxl / 2 + ddx - 1) / 2, y0 + (dyl + dyl / 2 + ddy - 1) / 2, dir)
                    self.render(x0 + ddx + (dxl + dxl / 2 + ddx - 1) / 2, y0 + ddy + (dyl + dyl / 2 + ddy - 1) / 2, dir)
                    self.render(x0 + ddx + (dxl / 2 + ddx - 1) / 2, y0 + ddy + (dyl / 2 + ddy - 1) / 2, dir)
                    self.render(x0 + 2 * ddx + (dxl / 2 + ddx - 1) / 2, y0 + 2 * ddy + (dyl / 2 + ddy - 1) / 2, dir)
                    self.render(x0 + 2 * ddx + (dxl + dxl / 2 + ddx - 1) / 2,
                                y0 + 2 * ddy + (dyl + dyl / 2 + ddy - 1) / 2, dir)
                    return

                if (abs(dxl + dyl) == 3):
                    ddx = dxl / abs(dxl + dyl)
                    ddy = dyl / abs(dxl + dyl)
                    self.render(x0 + (dxr / 2 + ddx - 1) / 2, y0 + (dyr / 2 + ddy - 1) / 2, dir)
                    self.render(x0 + (dxr + dxr / 2 + ddx - 1) / 2, y0 + (dyr + dyr / 2 + ddy - 1) / 2, dir)
                    self.render(x0 + ddx + (dxr + dxr / 2 + ddx - 1) / 2, y0 + ddy + (dyr + dyr / 2 + ddy - 1) / 2, dir)
                    self.render(x0 + ddx + (dxr / 2 + ddx - 1) / 2, y0 + ddy + (dyr / 2 + ddy - 1) / 2, dir)
                    self.render(x0 + 2 * ddx + (dxr / 2 + ddx - 1) / 2, y0 + 2 * ddy + (dyr / 2 + ddy - 1) / 2, dir)
                    self.render(x0 + 2 * ddx + (dxr + dxr / 2 + ddx - 1) / 2,
                                y0 + 2 * ddy + (dyr + dyr / 2 + ddy - 1) / 2, dir)
                    return

            return

        # divide into 2 parts if necessary
        if (2 * (abs(dxl) + abs(dyl)) > 3 * (abs(dxr) + abs(dyr))):  # left side much longer than right side
            # var dxl2=Math.round(dxl/2)
            # var dyl2=Math.round(dyl/2)
            dxl2 = round(dxl / 2)
            dyl2 = round(dyl / 2)
            if ((abs(dxr) + abs(dyr)) % 2 == 0):  # right side is even
                if ((abs(dxl) + abs(dyl)) % 2 == 0):  # make 2 parts from even side
                    if (dir == "l"):
                        if ((abs(dxl) + abs(dyl)) % 4 == 0):  # make 2 parts even-even from even side
                            self.go(x0, y0, dxl2, dyl2, dxr, dyr, "l")
                            self.go(x0 + dxl2, y0 + dyl2, dxl - dxl2, dyl - dyl2, dxr, dyr, "l")

                        else:  # make 2 parts odd-odd from even side
                            self.go(x0, y0, dxl2, dyl2, dxr, dyr, "m")
                            self.go(x0 + dxl2 + dxr, y0 + dyl2 + dyr, -dxr, -dyr, dxl - dxl2, dyl - dyl2, "m")

                        return


                else:  # make 2 parts from odd side
                    if (dir == "m"):
                        if ((abs(dxl2) + abs(dyl2)) % 2 == 0):
                            self.go(x0, y0, dxl2, dyl2, dxr, dyr, "l")
                            self.go(x0 + dxl2, y0 + dyl2, dxl - dxl2, dyl - dyl2, dxr, dyr, "m")
                        else:
                            self.go(x0, y0, dxl2, dyl2, dxr, dyr, "m")
                            self.go(x0 + dxl2 + dxr, y0 + dyl2 + dyr, -dxr, -dyr, dxl - dxl2, dyl - dyl2, "r")

                        return



            else:  # right side is odd
                if (dir == "l"):
                    self.go(x0, y0, dxl2, dyl2, dxr, dyr, "l")
                    self.go(x0 + dxl2, y0 + dyl2, dxl - dxl2, dyl - dyl2, dxr, dyr, "l")
                    return

                if (dir == "m"):
                    self.go(x0, y0, dxl2, dyl2, dxr, dyr, "l")
                    self.go(x0 + dxl2, y0 + dyl2, dxl - dxl2, dyl - dyl2, dxr, dyr, "m")
                    return

        if (2 * (abs(dxr) + abs(dyr)) > 3 * (abs(dxl) + abs(dyl))):  # right side much longer than left side
            # var dxr2=Math.round(dxr/2)
            # var dyr2=Math.round(dyr/2)
            dxr2 = round(dxr / 2)
            dyr2 = round(dyr / 2)
            if ((abs(dxl) + abs(dyl)) % 2 == 0):  # left side is even
                if ((abs(dxr) + abs(dyr)) % 2 == 0):  # make 2 parts from even side
                    if (dir == "r"):
                        if ((abs(dxr) + abs(dyr)) % 4 == 0):  # make 2 parts even-even from even side
                            self.go(x0, y0, dxl, dyl, dxr2, dyr2, "r")
                            self.go(x0 + dxr2, y0 + dyr2, dxl, dyl, dxr - dxr2, dyr - dyr2, "r")

                        else:  # make 2 parts odd-odd from even side
                            self.go(x0, y0, dxl, dyl, dxr2, dyr2, "m")
                            self.go(x0 + dxr2 + dxl, y0 + dyr2 + dyl, dxr - dxr2, dyr - dyr2, -dxl, -dyl, "m")

                        return


                else:  # make 2 parts from odd side
                    if (dir == "m"):
                        if ((abs(dxr2) + abs(dyr2)) % 2 == 0):
                            self.go(x0, y0, dxl, dyl, dxr2, dyr2, "r")
                            self.go(x0 + dxr2, y0 + dyr2, dxl, dyl, dxr - dxr2, dyr - dyr2, "m")

                        else:
                            self.go(x0, y0, dxl, dyl, dxr2, dyr2, "m")
                            self.go(x0 + dxr2 + dxl, y0 + dyr2 + dyl, dxr - dxr2, dyr - dyr2, -dxl, -dyl, "l")

                        return



            else:  # left side is odd
                if (dir == "r"):
                    self.go(x0, y0, dxl, dyl, dxr2, dyr2, "r")
                    self.go(x0 + dxr2, y0 + dyr2, dxl, dyl, dxr - dxr2, dyr - dyr2, "r")
                    return

                if (dir == "m"):
                    self.go(x0, y0, dxl, dyl, dxr2, dyr2, "r")
                    self.go(x0 + dxr2, y0 + dyr2, dxl, dyl, dxr - dxr2, dyr - dyr2, "m")
                    return

        # divide into 2x2 parts
        if ((dir == "l") or (dir == "r")):
            # var dxl2=Math.round(dxl/2)
            # var dyl2=Math.round(dyl/2)
            # var dxr2=Math.round(dxr/2)
            # var dyr2=Math.round(dyr/2)
            dxl2 = round(dxl / 2)
            dyl2 = round(dyl / 2)
            dxr2 = round(dxr / 2)
            dyr2 = round(dyr / 2)
            if ((abs(dxl + dyl) % 2 == 0) and (abs(dxr + dyr) % 2 == 0)):  # even-even
                if (abs(dxl2 + dyl2 + dxr2 + dyr2) % 2 == 0):  # ee-ee or oo-oo
                    if (dir == "l"):
                        self.go(x0, y0, dxl2, dyl2, dxr2, dyr2, "r")
                        self.go(x0 + dxr2, y0 + dyr2, dxl2, dyl2, dxr - dxr2, dyr - dyr2, "l")
                        self.go(x0 + dxr2 + dxl2, y0 + dyr2 + dyl2, dxl - dxl2, dyl - dyl2, dxr - dxr2, dyr - dyr2, "l")
                        self.go(x0 + dxr2 + dxl, y0 + dyr2 + dyl, dxl2 - dxl, dyl2 - dyl, -dxr2, -dyr2, "r")

                    else:
                        self.go(x0, y0, dxl2, dyl2, dxr2, dyr2, "l")
                        self.go(x0 + dxl2, y0 + dyl2, dxl - dxl2, dyl - dyl2, dxr2, dyr2, "r")
                        self.go(x0 + dxr2 + dxl2, y0 + dyr2 + dyl2, dxl - dxl2, dyl - dyl2, dxr - dxr2, dyr - dyr2, "r")
                        self.go(x0 + dxr + dxl2, y0 + dyr + dyl2, -dxl2, -dyl2, dxr2 - dxr, dyr2 - dyr, "l")


                else:  # ee-oo or oo-ee
                    if ((dxr2 + dyr2) % 2 == 0):  # ee-oo
                        if (dir == "l"):
                            self.go(x0, y0, dxl2, dyl2, dxr2, dyr2, "r")
                            self.go(x0 + dxr2, y0 + dyr2, dxl2, dyl2, dxr - dxr2, dyr - dyr2, "m")
                            self.go(x0 + dxr + dxl2, y0 + dyr + dyl2, dxr2 - dxr, dyr2 - dyr, dxl - dxl2, dyl - dyl2,
                                    "m")
                            self.go(x0 + dxr2 + dxl, y0 + dyr2 + dyl, dxl2 - dxl, dyl2 - dyl, -dxr2, -dyr2, "r")

                        else:  # ee-oo for dir="r" not possible, so transforming into e-1,e+1-oo = oo-oo
                            if (dxr2 != 0):
                                dxr2 += 1
                            else:
                                dyr2 += 1
                            self.go(x0, y0, dxl2, dyl2, dxr2, dyr2, "l")
                            self.go(x0 + dxl2, y0 + dyl2, dxl - dxl2, dyl - dyl2, dxr2, dyr2, "m")
                            self.go(x0 + dxl + dxr2, y0 + dyl + dyr2, dxr - dxr2, dyr - dyr2, dxl2 - dxl, dyl2 - dyl,
                                    "m")
                            self.go(x0 + dxl2 + dxr, y0 + dyl2 + dyr, -dxl2, -dyl2, dxr2 - dxr, dyr2 - dyr, "l")


                    else:  # oo-ee
                        if (dir == "r"):
                            self.go(x0, y0, dxl2, dyl2, dxr2, dyr2, "l")
                            self.go(x0 + dxl2, y0 + dyl2, dxl - dxl2, dyl - dyl2, dxr2, dyr2, "m")
                            self.go(x0 + dxl + dxr2, y0 + dyl + dyr2, dxr - dxr2, dyr - dyr2, dxl2 - dxl, dyl2 - dyl,
                                    "m")
                            self.go(x0 + dxl2 + dxr, y0 + dyl2 + dyr, -dxl2, -dyl2, dxr2 - dxr, dyr2 - dyr, "l")

                        else:  # oo-ee for dir="l" not possible, so transforming into oo-e-1,e+1 = oo-oo
                            if (dxl2 != 0):
                                dxl2 += 1
                            else:
                                dyl2 += 1
                            self.go(x0, y0, dxl2, dyl2, dxr2, dyr2, "r")
                            self.go(x0 + dxr2, y0 + dyr2, dxl2, dyl2, dxr - dxr2, dyr - dyr2, "m")
                            self.go(x0 + dxr + dxl2, y0 + dyr + dyl2, dxr2 - dxr, dyr2 - dyr, dxl - dxl2, dyl - dyl2,
                                    "m")
                            self.go(x0 + dxr2 + dxl, y0 + dyr2 + dyl, dxl2 - dxl, dyl2 - dyl, -dxr2, -dyr2, "r")




            else:  # not even-even
                if ((abs(dxl + dyl) % 2 != 0) and (abs(dxr + dyr) % 2 != 0)):  # odd-odd
                    if (dxl2 % 2 != 0): dxl2 = dxl - dxl2  # get it in a shape eo-eo
                    if (dyl2 % 2 != 0): dyl2 = dyl - dyl2
                    if (dxr2 % 2 != 0): dxr2 = dxr - dxr2
                    if (dyr2 % 2 != 0): dyr2 = dyr - dyr2
                    if (dir == "l"):
                        self.go(x0, y0, dxl2, dyl2, dxr2, dyr2, "r")
                        self.go(x0 + dxr2, y0 + dyr2, dxl2, dyl2, dxr - dxr2, dyr - dyr2, "m")
                        self.go(x0 + dxr + dxl2, y0 + dyr + dyl2, dxr2 - dxr, dyr2 - dyr, dxl - dxl2, dyl - dyl2, "m")
                        self.go(x0 + dxr2 + dxl, y0 + dyr2 + dyl, dxl2 - dxl, dyl2 - dyl, -dxr2, -dyr2, "r")

                    else:
                        self.go(x0, y0, dxl2, dyl2, dxr2, dyr2, "l")
                        self.go(x0 + dxl2, y0 + dyl2, dxl - dxl2, dyl - dyl2, dxr2, dyr2, "m")
                        self.go(x0 + dxl + dxr2, y0 + dyl + dyr2, dxr - dxr2, dyr - dyr2, dxl2 - dxl, dyl2 - dyl, "m")
                        self.go(x0 + dxl2 + dxr, y0 + dyl2 + dyr, -dxl2, -dyl2, dxr2 - dxr, dyr2 - dyr, "l")


                else:  # even-odd or odd-even
                    if (abs(dxl + dyl) % 2 == 0):  # odd-even
                        if (dir == "l"):
                            if (dxr2 % 2 != 0): dxr2 = dxr - dxr2  # get it in a shape eo-xx
                            if (dyr2 % 2 != 0): dyr2 = dyr - dyr2
                            if (abs(dxl + dyl) > 2):
                                self.go(x0, y0, dxl2, dyl2, dxr2, dyr2, "r")
                                self.go(x0 + dxr2, y0 + dyr2, dxl2, dyl2, dxr - dxr2, dyr - dyr2, "l")
                                self.go(x0 + dxr2 + dxl2, y0 + dyr2 + dyl2, dxl - dxl2, dyl - dyl2, dxr - dxr2,
                                        dyr - dyr2, "l")
                                self.go(x0 + dxr2 + dxl, y0 + dyr2 + dyl, dxl2 - dxl, dyl2 - dyl, -dxr2, -dyr2, "r")

                            else:
                                self.go(x0, y0, dxl2, dyl2, dxr2, dyr2, "r")
                                self.go(x0 + dxr2, y0 + dyr2, dxl2, dyl2, dxr - dxr2, dyr - dyr2, "m")
                                self.go(x0 + dxr + dxl2, y0 + dyr + dyl2, dxr2 - dxr, dyr2 - dyr, dxl - dxl2,
                                        dyl - dyl2, "m")
                                self.go(x0 + dxr2 + dxl, y0 + dyr2 + dyl, dxl2 - dxl, dyl2 - dyl, -dxr2, -dyr2, "r")


                        else:
                            print(
                                "4-part-error1: " + x0 + ", " + y0 + ", " + dxl + ", " + dyl + ", " + dxr + ", " + dyr + ", " + dir)

                    else:  # even-odd
                        if (dir == "r"):
                            if (dxl2 % 2 != 0): dxl2 = dxl - dxl2  # get it in a shape xx-eo
                            if (dyl2 % 2 != 0): dyl2 = dyl - dyl2
                            if (abs(dxr + dyr) > 2):
                                self.go(x0, y0, dxl2, dyl2, dxr2, dyr2, "l")
                                self.go(x0 + dxl2, y0 + dyl2, dxl - dxl2, dyl - dyl2, dxr2, dyr2, "r")
                                self.go(x0 + dxr2 + dxl2, y0 + dyr2 + dyl2, dxl - dxl2, dyl - dyl2, dxr - dxr2,
                                        dyr - dyr2, "r")
                                self.go(x0 + dxr + dxl2, y0 + dyr + dyl2, -dxl2, -dyl2, dxr2 - dxr, dyr2 - dyr, "l")

                            else:
                                self.go(x0, y0, dxl2, dyl2, dxr2, dyr2, "l")
                                self.go(x0 + dxl2, y0 + dyl2, dxl - dxl2, dyl - dyl2, dxr2, dyr2, "m")
                                self.go(x0 + dxl + dxr2, y0 + dyl + dyr2, dxr - dxr2, dyr - dyr2, dxl2 - dxl,
                                        dyl2 - dyl, "m")
                                self.go(x0 + dxl2 + dxr, y0 + dyl2 + dyr, -dxl2, -dyl2, dxr2 - dxr, dyr2 - dyr, "l")


                        else:
                            print(
                                "4-part-error2: " + x0 + ", " + y0 + ", " + dxl + ", " + dyl + ", " + dxr + ", " + dyr + ", " + dir)




        else:  # dir=="m" -> divide into 3x3 parts
            if ((abs(dxl + dyl) % 2 == 0) and (abs(dxr + dyr) % 2 == 0)):
                print(
                    "9-part-error1: " + x0 + ", " + y0 + ", " + dxl + ", " + dyl + ", " + dxr + ", " + dyr + ", " + dir)
            if (abs(dxr + dyr) % 2 == 0):  # even-odd: oeo-ooo
                # var dxl2=Math.round(dxl/3)
                # var dyl2=Math.round(dyl/3)
                # var dxr2=Math.round(dxr/3)
                # var dyr2=Math.round(dyr/3)
                dxl2 = round(dxl / 3)
                dyl2 = round(dyl / 3)
                dxr2 = round(dxr / 3)
                dyr2 = round(dyr / 3)
                if ((dxl2 + dyl2) % 2 == 0):  # make it odd
                    dxl2 = dxl - 2 * dxl2
                    dyl2 = dyl - 2 * dyl2

                if ((
                        dxr2 + dyr2) % 2 == 0):  # make it odd (not necessary, however results are better for 12x30, 18x30 etc.)
                    if (abs(dxr2 + dyr2) != 2):
                        if (dxr < 0): dxr2 += 1
                        if (dxr > 0): dxr2 -= 1  # dont use else here !
                        if (dyr < 0): dyr2 += 1
                        if (dyr > 0): dyr2 -= 1  # dont use else here !



            else:  # odd-even: ooo-oeo
                # var dxl2=Math.round(dxl/3)
                # var dyl2=Math.round(dyl/3)
                # var dxr2=Math.round(dxr/3)
                # var dyr2=Math.round(dyr/3)
                dxl2 = round(dxl / 3)
                dyl2 = round(dyl / 3)
                dxr2 = round(dxr / 3)
                dyr2 = round(dyr / 3)
                if ((dxr2 + dyr2) % 2 == 0):  # make it odd
                    dxr2 = dxr - 2 * dxr2
                    dyr2 = dyr - 2 * dyr2

                if ((
                        dxl2 + dyl2) % 2 == 0):  # make it odd (not necessary, however results are better for 12x30, 18x30 etc.)
                    if (abs(dxl2 + dyl2) != 2):
                        if (dxl < 0): dxl2 += 1
                        if (dxl > 0): dxl2 -= 1  # dont use else here !
                        if (dyl < 0): dyl2 += 1
                        if (dyl > 0): dyl2 -= 1  # dont use else here !

            if (abs(dxl + dyl) < abs(dxr + dyr)):
                self.go(x0, y0, dxl2, dyl2, dxr2, dyr2, "m")
                self.go(x0 + dxl2 + dxr2, y0 + dyl2 + dyr2, -dxr2, -dyr2, dxl - 2 * dxl2, dyl - 2 * dyl2, "m")
                self.go(x0 + dxl - dxl2, y0 + dyl - dyl2, dxl2, dyl2, dxr2, dyr2, "m")
                self.go(x0 + dxl + dxr2, y0 + dyl + dyr2, dxr - 2 * dxr2, dyr - 2 * dyr2, -dxl2, -dyl2, "m")
                self.go(x0 + dxr - dxr2 + dxl - dxl2, y0 + dyr - dyr2 + dyl - dyl2, 2 * dxl2 - dxl, 2 * dyl2 - dyl,
                        2 * dxr2 - dxr, 2 * dyr2 - dyr, "m")
                self.go(x0 + dxl2 + dxr2, y0 + dyl2 + dyr2, dxr - 2 * dxr2, dyr - 2 * dyr2, -dxl2, -dyl2, "m")
                self.go(x0 + dxr - dxr2, y0 + dyr - dyr2, dxl2, dyl2, dxr2, dyr2, "m")
                self.go(x0 + dxr + dxl2, y0 + dyr + dyl2, -dxr2, -dyr2, dxl - 2 * dxl2, dyl - 2 * dyl2, "m")
                self.go(x0 + dxr - dxr2 + dxl - dxl2, y0 + dyr - dyr2 + dyl - dyl2, dxl2, dyl2, dxr2, dyr2, "m")

            else:
                self.go(x0, y0, dxl2, dyl2, dxr2, dyr2, "m")
                self.go(x0 + dxl2 + dxr2, y0 + dyl2 + dyr2, dxr - 2 * dxr2, dyr - 2 * dyr2, -dxl2, -dyl2, "m")
                self.go(x0 + dxr - dxr2, y0 + dyr - dyr2, dxl2, dyl2, dxr2, dyr2, "m")
                self.go(x0 + dxr + dxl2, y0 + dyr + dyl2, -dxr2, -dyr2, dxl - 2 * dxl2, dyl - 2 * dyl2, "m")
                self.go(x0 + dxr - dxr2 + dxl - dxl2, y0 + dyr - dyr2 + dyl - dyl2, 2 * dxl2 - dxl, 2 * dyl2 - dyl,
                        2 * dxr2 - dxr, 2 * dyr2 - dyr, "m")
                self.go(x0 + dxl2 + dxr2, y0 + dyl2 + dyr2, -dxr2, -dyr2, dxl - 2 * dxl2, dyl - 2 * dyl2, "m")
                self.go(x0 + dxl - dxl2, y0 + dyl - dyl2, dxl2, dyl2, dxr2, dyr2, "m")
                self.go(x0 + dxl + dxr2, y0 + dyl + dyr2, dxr - 2 * dxr2, dyr - 2 * dyr2, -dxl2, -dyl2, "m")
                self.go(x0 + dxr - dxr2 + dxl - dxl2, y0 + dyr - dyr2 + dyl - dyl2, dxl2, dyl2, dxr2, dyr2, "m")

    def plot_ordered_curve(self):
        figure, ax = plt.subplots()

        x = self.ordered_points[:, 0]
        y = self.ordered_points[:, 1]

        num_points = len(x)
        order = np.arange(num_points)  # 创建顺序数组

        # 创建颜色渐变
        cmap = plt.get_cmap('plasma')  # 选择颜色映射，这里使用 'plasma'，你可以根据需要选择其他颜色映射
        colors = cmap(order / num_points)  # 使用出现顺序归一化作为颜色映射

        # 绘制颜色渐变的点
        plt.scatter(x, y, c=colors, cmap='plasma', edgecolor='none')
        plt.colorbar(label='Order')  # 添加颜色条，并标注代表的含义
        plt.title('Color Gradient Scatter Plot')
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.invert_yaxis()
        plt.show()

    def plot_curve(self):
        figure, ax = plt.subplots()

        x1, y1 = self.seg_list[0, 0], self.seg_list[0, 1]
        for x2, y2 in self.seg_list[1:, :]:
            ax.plot([x1, x2], [y1, y2])
            x1 = x2
            y1 = y2

        ax.set_aspect(1)
        ax.invert_yaxis()
        figure.show()
        return


class HSCMapper(object):

    def __init__(self,
                 chw,
                 **kwargs):
        self.channel, self.height, self.width = chw
        # self.n2h = []   # neuron-id to hsc-id
        self.h2n = None   # hsc-id to neuron-id
        self.hsc = HSCurve(self.width, self.height)

    def build(self, channel_first=False):
        self.hsc.compute_sfc()
        if channel_first:
            # CHW, flatten in wh-channel order
            y = self.hsc.ordered_points[:, 0].reshape(-1)
            x = self.hsc.ordered_points[:, 1].reshape(-1)
            h2n = x + y * self.width
            h2n = np.tile(h2n, (self.channel, 1))
            self.h2n = (h2n + np.arange(self.channel).reshape(-1, 1) * self.width * self.height).reshape(-1)
        else:
            # HWC, flatten in channel-wh order
            y = self.hsc.ordered_points[:, 0].reshape(-1)
            x = self.hsc.ordered_points[:, 1].reshape(-1)
            h2n = x + y * self.width
            h2n = h2n.reshape(-1, 1) + np.arange(self.channel) * self.width * self.height
            self.h2n = h2n.reshape(-1)

    def draw(self, title='title'):
        figure, ax = plt.subplots()
        c = self.h2n // (self.width * self.height)
        y = (self.h2n - c * self.width * self.height) // self.width
        x = self.h2n - c * self.width * self.height - y * self.width
        x = x + c * self.width

        num_points = len(x)
        order = np.arange(num_points)  # 创建顺序数组
        # 创建颜色渐变
        cmap = plt.get_cmap('plasma')  # 选择颜色映射，这里使用 'plasma'，你可以根据需要选择其他颜色映射
        colors = cmap(order / num_points)  # 使用出现顺序归一化作为颜色映射

        # 绘制颜色渐变的点
        plt.scatter(x, y, c=colors, cmap='plasma', edgecolor='none')
        plt.colorbar(label='Order')  # 添加颜色条，并标注代表的含义
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.invert_yaxis()
        plt.show()

